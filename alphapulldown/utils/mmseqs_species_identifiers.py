"""Helpers for recovering species identifiers from mmseqs-derived A3Ms."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence
from urllib import error
from urllib import parse
from urllib import request

from absl import logging
from alphafold.data import parsers
import numpy as np

_UNIPROT_HEADER_PATTERN = re.compile(
    r"""
    ^
    (?:tr|sp)
    \|
    (?P<accession>[A-Za-z0-9]{6,10})
    (?:_\d)?
    \|
    (?:[A-Za-z0-9]+)
    _
    (?P<species>[A-Za-z0-9]{1,5})
    (?:_\d+)?
    $
    """,
    re.VERBOSE,
)
_UNIREF_HEADER_PATTERN = re.compile(
    r'^UniRef\d+_(?P<accession>[A-Za-z0-9]+)$'
)
_UNIPARC_HEADER_PATTERN = re.compile(r'^(?P<accession>UPI[A-Z0-9]+)$')
_UNIPROT_ACCESSION_PATTERN = re.compile(
    r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$'
)

# UniProtKB taxa are fetched with the /uniprotkb/accessions endpoint, which
# returns known entries directly instead of running a search, so it takes much
# larger batches than the former "accession:A OR accession:B" search query.
_UNIPROT_BATCH_SIZE = 100
# Batches sent at the same time by one process. Kept at 1: feature generation
# usually runs many processes in parallel already, and concurrent large
# requests make UniProt slow down until they time out.
_UNIPROT_MAX_PARALLEL_BATCHES = 1
# Extra attempts for a batch on HTTP 429 / 5xx before it is split in two.
_UNIPROT_HTTP_RETRIES = 2
_UNIPROT_RETRY_DELAY_SECONDS = 2.0
_UNIPARC_BATCH_SIZE = 16
_UNIPROT_TIMEOUT_SECONDS = 60
# A batch that times out is retried once, then split in two (once).
_UNIPROT_TIMEOUT_RETRIES = 1
_SPECIES_ID_CACHE: dict[str, str] = {}
_IDENTIFIER_CACHE_KEYS = (
    "source_sha256",
    "msa_species_identifiers",
    "msa_uniprot_accession_identifiers",
)

# Persistent cache (accession -> taxid) shared by every process and every run,
# so an accession is fetched from UniProt once instead of once per protein.
# Location: $ALPHAPULLDOWN_SPECIES_CACHE if set, otherwise the default below.
# Set the variable to "none" to disable it; delete the file to start over.
_PERSISTENT_CACHE_ENV = "ALPHAPULLDOWN_SPECIES_CACHE"
_PERSISTENT_CACHE_DEFAULT = os.path.join(
    "~", ".cache", "alphapulldown", "uniprot_species_cache.sqlite"
)
_PERSISTENT_CACHE_TIMEOUT_SECONDS = 60
_SQLITE_MAX_PARAMS = 500


def _extract_sequence_identifier(description: str) -> str:
    split_description = description.split()
    if not split_description:
        return ""
    return split_description[0].partition('/')[0]


def _extract_accession_and_species(description: str) -> tuple[str, str]:
    sequence_identifier = _extract_sequence_identifier(description)
    if not sequence_identifier:
        return "", ""

    matches = _UNIPROT_HEADER_PATTERN.search(sequence_identifier.strip())
    if matches:
        return matches.group("accession"), matches.group("species")

    matches = _UNIREF_HEADER_PATTERN.search(sequence_identifier.strip())
    if matches:
        accession = matches.group("accession")
        if _is_resolvable_accession(accession):
            return accession, ""
        return "", ""

    matches = _UNIPARC_HEADER_PATTERN.search(sequence_identifier.strip())
    if matches:
        return matches.group("accession"), ""

    if _UNIPROT_ACCESSION_PATTERN.search(sequence_identifier.strip()):
        return sequence_identifier.strip(), ""

    return "", ""


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _is_transport_error(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError, error.URLError)) and not isinstance(
        exc, error.HTTPError
    )


def _is_resolvable_accession(accession: str) -> bool:
    return accession.startswith("UPI") or bool(
        _UNIPROT_ACCESSION_PATTERN.fullmatch(accession)
    )


def _query_uniprot_batch(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> dict[str, object]:
    url = (
        "https://rest.uniprot.org/uniprotkb/accessions?accessions="
        f"{parse.quote(','.join(accessions), safe=',')}"
        "&fields=accession,organism_id"
        "&format=json"
        f"&size={len(accessions)}"
    )
    with urlopen(url, timeout=_UNIPROT_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _query_uniparc_batch(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> dict[str, object]:
    query = " OR ".join(f"upi:{accession}" for accession in accessions)
    url = (
        "https://rest.uniprot.org/uniparc/search?query="
        f"{parse.quote(query)}"
        "&fields=upi,organism_id"
        "&format=json"
        f"&size={len(accessions)}"
    )
    with urlopen(url, timeout=_UNIPROT_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _is_retryable_http_error(exc: Exception) -> bool:
    return isinstance(exc, error.HTTPError) and (exc.code == 429 or exc.code >= 500)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):  # socket.timeout, SSL read timeouts
        return True
    if isinstance(exc, error.URLError) and isinstance(exc.reason, TimeoutError):
        return True
    return _is_transport_error(exc) and "timed out" in str(exc).lower()


def _query_uniprot_batch_resilient(
    batch: Sequence[str],
    *,
    urlopen: Callable[..., object],
    retries: int = _UNIPROT_HTTP_RETRIES,
    timeout_retries: int = _UNIPROT_TIMEOUT_RETRIES,
    split_on_timeout: bool = True,
) -> tuple[list[dict[str, object]], set[str]]:
    """Query one UniProtKB batch; return (results, accessions UniProt answered).

    - a timeout (UniProt overloaded) is retried once, then the batch is split
      in two once: smaller requests answer faster. The cost stays bounded
      (at most 4 requests per batch) even when UniProt is down;
    - other network (transport) errors give up on the batch;
    - unanswered accessions are never cached, so they are retried next time;
    - HTTP 429/5xx are retried a few times with a growing delay;
    - an HTTP error that persists splits the batch in two (halves are not
      retried again), so one problematic accession no longer costs one
      request per accession of its batch;
    - an accession that still fails on its own is left unanswered (never
      cached), so a rejected request cannot store empty species.
    """
    attempt = 0
    while True:
        try:
            payload = _query_uniprot_batch(batch, urlopen=urlopen)
            return list(payload.get("results", [])), set(batch)
        except Exception as exc:  # pragma: no cover - best-effort network fallback
            if _is_timeout_error(exc):
                if attempt < timeout_retries:
                    attempt += 1
                    time.sleep(_UNIPROT_RETRY_DELAY_SECONDS * attempt)
                    continue
                logging.warning(
                    "Unable to resolve UniProtKB taxonomy for %d accessions: %s",
                    len(batch),
                    exc,
                )
                if not split_on_timeout or len(batch) == 1:
                    return [], set()
                middle = len(batch) // 2
                halves = [
                    _query_uniprot_batch_resilient(
                        half,
                        urlopen=urlopen,
                        retries=0,
                        timeout_retries=0,
                        split_on_timeout=False,
                    )
                    for half in (batch[:middle], batch[middle:])
                ]
                return halves[0][0] + halves[1][0], halves[0][1] | halves[1][1]
            if _is_transport_error(exc):
                logging.warning(
                    "Unable to resolve UniProtKB taxonomy for %d accessions: %s",
                    len(batch),
                    exc,
                )
                return [], set()
            if _is_retryable_http_error(exc) and attempt < retries:
                attempt += 1
                time.sleep(_UNIPROT_RETRY_DELAY_SECONDS * attempt)
                continue
            logging.warning(
                "Unable to resolve UniProtKB taxonomy for %d accessions: %s",
                len(batch),
                exc,
            )
            if len(batch) == 1:
                return [], set()
            middle = len(batch) // 2
            left_results, left_answered = _query_uniprot_batch_resilient(
                batch[:middle], urlopen=urlopen, retries=0, timeout_retries=0,
                split_on_timeout=False,
            )
            right_results, right_answered = _query_uniprot_batch_resilient(
                batch[middle:], urlopen=urlopen, retries=0, timeout_retries=0,
                split_on_timeout=False,
            )
            return left_results + right_results, left_answered | right_answered


def _query_uniprot_species_ids(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> tuple[dict[str, str], set[str]]:
    resolved: dict[str, str] = {}
    cacheable_misses: set[str] = set()
    requested = set(accessions)
    batches = list(_batched(sorted(requested), _UNIPROT_BATCH_SIZE))
    workers = min(_UNIPROT_MAX_PARALLEL_BATCHES, len(batches))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(
                executor.map(
                    lambda batch: _query_uniprot_batch_resilient(batch, urlopen=urlopen),
                    batches,
                )
            )
    else:
        outcomes = [
            _query_uniprot_batch_resilient(batch, urlopen=urlopen) for batch in batches
        ]
    unanswered = len(requested) - sum(len(answered) for _, answered in outcomes)
    if unanswered:
        logging.warning(
            "%d UniProtKB accessions left without species after network errors; "
            "they will be queried again next time.",
            unanswered,
        )
    for results, answered in outcomes:
        cacheable_misses.update(answered)
        for result in results:
            taxon_id = (result.get("organism") or {}).get("taxonId")
            if taxon_id is None:
                continue  # e.g. inactive (deleted) entries carry no organism
            accession = result.get("primaryAccession")
            if accession:
                resolved[accession] = str(taxon_id)
    return resolved, cacheable_misses


def _query_uniparc_species_ids(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> tuple[dict[str, str], set[str]]:
    resolved: dict[str, str] = {}
    cacheable_misses: set[str] = set()
    for batch in _batched(sorted(set(accessions)), _UNIPARC_BATCH_SIZE):
        try:
            payload = _query_uniparc_batch(batch, urlopen=urlopen)
            cacheable_misses.update(batch)
        except Exception as exc:  # pragma: no cover - best-effort network fallback
            logging.warning(
                "Unable to resolve UniParc taxonomy for %d accessions: %s",
                len(batch),
                exc,
            )
            if _is_transport_error(exc):
                continue
            payload = {"results": []}
            for accession in batch:
                try:
                    single_payload = _query_uniparc_batch([accession], urlopen=urlopen)
                except Exception as single_exc:
                    if _is_transport_error(single_exc):
                        continue
                    continue
                cacheable_misses.add(accession)
                payload["results"].extend(single_payload["results"])
        for result in payload.get("results", []):
            accession = result.get("uniParcId")
            organisms = result.get("organisms", [])
            taxon_ids = {
                str(organism.get("taxonId"))
                for organism in organisms
                if organism.get("taxonId") is not None
            }
            if accession and len(taxon_ids) == 1:
                resolved[accession] = next(iter(taxon_ids))
    return resolved, cacheable_misses


def _persistent_cache_path() -> str | None:
    """Return the shared cache file, or None when it is disabled."""
    value = os.environ.get(_PERSISTENT_CACHE_ENV, "").strip()
    if value.lower() in ("none", "off", "false", "0"):
        return None
    return os.path.abspath(os.path.expanduser(value or _PERSISTENT_CACHE_DEFAULT))


def _open_persistent_cache(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path, timeout=_PERSISTENT_CACHE_TIMEOUT_SECONDS)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass  # e.g. network filesystems: keep the default journal
    connection.execute(
        "CREATE TABLE IF NOT EXISTS species_ids ("
        "accession TEXT PRIMARY KEY, taxid TEXT NOT NULL)"
    )
    return connection


def _load_from_persistent_cache(accessions: Sequence[str]) -> None:
    """Copy the taxids known on disk into the in-memory cache (best effort)."""
    path = _persistent_cache_path()
    wanted = sorted(
        {a for a in accessions if a and a not in _SPECIES_ID_CACHE}
    )
    if not path or not wanted:
        return
    try:
        connection = _open_persistent_cache(path)
        try:
            for chunk in _batched(wanted, _SQLITE_MAX_PARAMS):
                placeholders = ",".join("?" * len(chunk))
                rows = connection.execute(
                    "SELECT accession, taxid FROM species_ids "
                    f"WHERE accession IN ({placeholders})",
                    list(chunk),
                )
                for accession, taxid in rows:
                    _SPECIES_ID_CACHE[accession] = taxid
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as exc:
        logging.warning("Unable to read species cache %s: %s", path, exc)


def _save_to_persistent_cache(entries: dict[str, str]) -> None:
    """Store accession -> taxid on disk (best effort).

    '' means UniProt answered but gave no species (e.g. deleted entry).
    A known taxid is never overwritten by an empty one.
    """
    path = _persistent_cache_path()
    if not path or not entries:
        return
    try:
        connection = _open_persistent_cache(path)
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO species_ids (accession, taxid) VALUES (?, ?) "
                    "ON CONFLICT(accession) DO UPDATE SET taxid = excluded.taxid "
                    "WHERE excluded.taxid != ''",
                    sorted(entries.items()),
                )
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as exc:
        logging.warning("Unable to write species cache %s: %s", path, exc)


# =================== Local species database (no UniProt) ===================
# Accession -> taxid index built once from the taxonomy ColabFold ships with
# uniref30_2302 by the separate script alphapulldown/scripts/
# build_local_species_db.py (installed as build_local_species_db.py). When
# $ALPHAPULLDOWN_LOCAL_SPECIES_DB names this file, species are read from it
# instead of the UniProt REST API. Accessions it does not know (e.g. the
# metagenomic colabfold_envdb identifiers) get no species, unless
# $ALPHAPULLDOWN_SPECIES_ONLINE_FALLBACK=1 sends them to UniProt.
_LOCAL_SPECIES_DB_ENV = "ALPHAPULLDOWN_LOCAL_SPECIES_DB"
_ONLINE_FALLBACK_ENV = "ALPHAPULLDOWN_SPECIES_ONLINE_FALLBACK"
_LOCAL_SPECIES_DB_TABLE = "local_species_ids"
_local_db_connections: dict[str, sqlite3.Connection] = {}
_local_db_warned: set[str] = set()


def _local_species_db_path() -> str | None:
    value = os.environ.get(_LOCAL_SPECIES_DB_ENV, "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else None


def _online_fallback_enabled() -> bool:
    if _local_species_db_path() is None:
        return True  # no local database configured: UniProt as before
    value = os.environ.get(_ONLINE_FALLBACK_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _local_species_db() -> sqlite3.Connection | None:
    """Read-only connection to the local species database, or None."""
    path = _local_species_db_path()
    if path is None:
        return None
    connection = _local_db_connections.get(path)
    if connection is not None:
        return connection
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False
        )
        connection.execute(f"SELECT 1 FROM {_LOCAL_SPECIES_DB_TABLE} LIMIT 1")
    except sqlite3.Error as exc:
        if path not in _local_db_warned:
            _local_db_warned.add(path)
            logging.warning(
                "Local species database %s is unusable (%s); falling back to "
                "UniProt. Build it with: build_local_species_db.py "
                "--colabfold-db-dir <dir> --out %s",
                path, exc, path,
            )
        return None
    _local_db_connections[path] = connection
    
    return connection


def _load_from_local_species_db(accessions: Sequence[str]) -> set[str]:
    """Put taxids known locally into the in-memory cache.

    Returns the accessions that were looked up (i.e. the database was usable),
    found or not.
    """
    connection = _local_species_db()
    wanted = sorted({a for a in accessions if a and a not in _SPECIES_ID_CACHE})
    if connection is None or not wanted:
        return set()
    try:
        for chunk in _batched(wanted, _SQLITE_MAX_PARAMS):
            rows = connection.execute(
                f"SELECT accession, taxid FROM {_LOCAL_SPECIES_DB_TABLE} "
                f"WHERE accession IN ({','.join('?' * len(chunk))})",
                list(chunk),
            )
            for accession, taxid in rows:
                _SPECIES_ID_CACHE[accession] = str(taxid)
    except sqlite3.Error as exc:
        logging.warning("Unable to read local species database: %s", exc)
        return set()
    found = sum(1 for accession in wanted if _SPECIES_ID_CACHE.get(accession))
    
    return set(wanted)


def resolve_species_ids_by_accession(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object] = request.urlopen,
) -> dict[str, str]:
    looked_up_locally = _load_from_local_species_db(
        [accession for accession in accessions if accession and _is_resolvable_accession(accession)]
    )
    if looked_up_locally and not _online_fallback_enabled():
        # Unknown to the local database: no species, and no UniProt query.
        for accession in looked_up_locally:
            _SPECIES_ID_CACHE.setdefault(accession, "")
    _load_from_persistent_cache(
        [accession for accession in accessions if accession and _is_resolvable_accession(accession)]
    )
    unresolved = [
        accession
        for accession in sorted(set(accessions))
        if accession
        and accession not in _SPECIES_ID_CACHE
        and _is_resolvable_accession(accession)
    ]
    for accession in accessions:
        if accession and accession not in _SPECIES_ID_CACHE and not _is_resolvable_accession(accession):
            _SPECIES_ID_CACHE.setdefault(accession, "")
    if unresolved:
        uniprot_accessions = [
            accession for accession in unresolved if not accession.startswith("UPI")
        ]
        uniparc_accessions = [
            accession for accession in unresolved if accession.startswith("UPI")
        ]
        resolved, cacheable_misses = _query_uniprot_species_ids(
            uniprot_accessions, urlopen=urlopen
        )
        uniparc_resolved, uniparc_cacheable_misses = _query_uniparc_species_ids(
            uniparc_accessions, urlopen=urlopen
        )
        resolved.update(uniparc_resolved)
        cacheable_misses.update(uniparc_cacheable_misses)
        for accession, species_id in resolved.items():
            _SPECIES_ID_CACHE[accession] = species_id
        for accession in cacheable_misses:
            _SPECIES_ID_CACHE.setdefault(accession, "")
        # Only accessions UniProt actually answered are stored; network
        # failures are left out so they are retried next time.
        _save_to_persistent_cache(
            {
                accession: _SPECIES_ID_CACHE[accession]
                for accession in set(resolved) | cacheable_misses
            }
        )

    return {
        accession: _SPECIES_ID_CACHE.get(accession, "")
        for accession in accessions
        if accession
    }


def strip_mmseq_comment_lines(a3m_string: str) -> str:
    return "\n".join(
        line for line in a3m_string.splitlines() if not line.startswith("#")
    )


def _calculate_mmseq_source_sha256(a3m_string: str) -> str:
    return hashlib.sha256(a3m_string.encode("utf-8")).hexdigest()


def _extract_mmseq_identifier_rows(a3m_string: str) -> tuple[list[str], list[str]]:
    msa = parsers.parse_a3m(a3m_string)
    seen_sequences: set[str] = set()
    accessions: list[str] = []
    species_ids: list[str] = []

    for sequence, description in zip(
        msa.sequences, msa.descriptions, strict=True
    ):
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        accession_id, species_id = _extract_accession_and_species(description)
        accessions.append(accession_id)
        species_ids.append(species_id)

    return accessions, species_ids


def _resolve_mmseq_identifier_rows(
    accessions: Sequence[str],
    species_ids: Sequence[str],
    *,
    species_resolver: Callable[[Sequence[str]], dict[str, str]],
) -> tuple[list[str], list[str]]:
    resolved_species_ids = species_resolver(
        [
            accession
            for accession, species_id in zip(accessions, species_ids, strict=True)
            if accession and not species_id
        ]
    )
    return list(accessions), [
        species_id or resolved_species_ids.get(accession_id, "")
        for accession_id, species_id in zip(accessions, species_ids, strict=True)
    ]


def _build_identifier_feature_arrays(
    accessions: Sequence[str],
    species_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        "msa_species_identifiers": np.array(
            [species_id.encode("utf-8") for species_id in species_ids],
            dtype=np.object_,
        ),
        "msa_uniprot_accession_identifiers": np.array(
            [accession_id.encode("utf-8") for accession_id in accessions],
            dtype=np.object_,
        ),
    }


def _load_identifier_rows_from_cache(
    cache_path: str,
    *,
    source_sha256: str,
    expected_rows: int | None,
) -> tuple[list[str], list[str]] | None:
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logging.warning(
            "Unable to read mmseqs identifier cache %s: %s", cache_path, exc
        )
        return None

    if not isinstance(payload, dict):
        return None
    if any(key not in payload for key in _IDENTIFIER_CACHE_KEYS):
        return None
    if payload["source_sha256"] != source_sha256:
        return None

    species_ids = payload["msa_species_identifiers"]
    accessions = payload["msa_uniprot_accession_identifiers"]
    if not isinstance(species_ids, list) or not isinstance(accessions, list):
        return None
    if len(species_ids) != len(accessions):
        return None
    if expected_rows is not None and len(species_ids) != expected_rows:
        return None
    if not all(isinstance(value, str) for value in species_ids + accessions):
        return None
    return accessions, species_ids


def _write_identifier_rows_to_cache(
    cache_path: str,
    *,
    source_sha256: str,
    accessions: Sequence[str],
    species_ids: Sequence[str],
) -> None:
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "source_sha256": source_sha256,
        "msa_species_identifiers": list(species_ids),
        "msa_uniprot_accession_identifiers": list(accessions),
    }
    fd, temp_path = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{os.path.basename(cache_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def build_mmseq_identifier_features(
    a3m_string: str,
    *,
    species_resolver: Callable[[Sequence[str]], dict[str, str]] | None = None,
    cache_path: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, np.ndarray]:
    if species_resolver is None:
        species_resolver = resolve_species_ids_by_accession
    stripped_a3m = strip_mmseq_comment_lines(a3m_string)
    source_sha256 = _calculate_mmseq_source_sha256(stripped_a3m)

    cached_rows = None
    if cache_path is not None:
        cached_rows = _load_identifier_rows_from_cache(
            cache_path,
            source_sha256=source_sha256,
            expected_rows=expected_rows,
        )

    if cached_rows is None:
        accessions, species_ids = _extract_mmseq_identifier_rows(stripped_a3m)
        accessions, species_ids = _resolve_mmseq_identifier_rows(
            accessions,
            species_ids,
            species_resolver=species_resolver,
        )
        if cache_path is not None and (
            expected_rows is None or len(species_ids) == expected_rows
        ):
            _write_identifier_rows_to_cache(
                cache_path,
                source_sha256=source_sha256,
                accessions=accessions,
                species_ids=species_ids,
            )
    else:
        accessions, species_ids = cached_rows

    return _build_identifier_feature_arrays(accessions, species_ids)


def enrich_mmseq_feature_dict_with_identifiers(
    feature_dict: dict[str, np.ndarray],
    a3m_string: str,
    *,
    species_resolver: Callable[[Sequence[str]], dict[str, str]] | None = None,
    cache_path: str | None = None,
) -> None:
    msa = feature_dict.get("msa")
    if msa is None:
        return
    identifier_features = build_mmseq_identifier_features(
        a3m_string,
        species_resolver=species_resolver,
        cache_path=cache_path,
        expected_rows=msa.shape[0],
    )
    if len(identifier_features["msa_species_identifiers"]) != msa.shape[0]:
        logging.warning(
            "Skipping mmseqs species enrichment because identifier rows do not "
            "match MSA rows: %d != %d",
            len(identifier_features["msa_species_identifiers"]),
            msa.shape[0],
        )
        return
    feature_dict.update(identifier_features)
