""" Implements structure prediction backend using AlphaFold.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
            Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de> 
            Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
import contextlib
import gc
import time
import json
import pickle
import subprocess
import sys
import enum
import typing
from collections.abc import Mapping
from typing import Dict, Union, List, Any
import os
from absl import logging
import numpy as np
from copy import copy
import jax.numpy as jnp
from alphapulldown.utils.plotting import plot_pae_from_matrix
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.post_modelling import post_prediction_process
#from alphapulldown.utils.calculate_rmsd import calculate_rmsd_and_superpose
from alphapulldown.utils.modelling_setup import pad_input_features
from alphapulldown.prediction.jax_compilation_cache import enable_persistent_compilation_cache
from alphapulldown.utils.af2_to_af3_msa import msa_rows_and_deletions_to_a3m
from alphafold.relax import relax
from alphafold.common import protein, residue_constants, confidence
from .folding_backend import FoldingBackend
logging.set_verbosity(logging.INFO)


# Relaxation parameters.
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
# Rows of the PAE computed at a time: bounds the probabilities held in memory to
# PAE_ROW_CHUNK_SIZE x N x bins instead of N x N x bins.
PAE_ROW_CHUNK_SIZE = 128


def _select_models_to_relax(
    ranked_order: List[str],
    *,
    models_to_relax: "ModelsToRelax",
    iptm_scores: Dict[str, float],
    ptm_scores: Dict[str, float],
    relax_best_score_threshold,
) -> List[str]:
    if models_to_relax == ModelsToRelax.ALL:
        return ranked_order
    if models_to_relax == ModelsToRelax.NONE:
        return []
    if not ranked_order:
        return []

    best_model = ranked_order[0]
    if relax_best_score_threshold is None:
        return [best_model]

    if best_model in iptm_scores:
        score_name = "iPTM"
        score_value = iptm_scores[best_model]
    else:
        score_name = "pTM"
        score_value = ptm_scores.get(best_model)

    if score_value is None or score_value < relax_best_score_threshold:
        logging.info(
            "Skipping relaxation for %s because its %s score %.3f is below the "
            "requested threshold %.3f.",
            best_model,
            score_name,
            0.0 if score_value is None else score_value,
            relax_best_score_threshold,
        )
        return []

    return [best_model]


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2


def _ensure_typing_dataclass_transform() -> None:
    """Provide typing.dataclass_transform on Python versions that lack it."""
    if hasattr(typing, "dataclass_transform"):
        return

    from typing_extensions import dataclass_transform

    typing.dataclass_transform = dataclass_transform


def _get_openmm_platform_names() -> List[str]:
    """Return the available OpenMM platform names."""
    try:
        import openmm
    except ImportError:
        return []

    try:
        return [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to inspect OpenMM platforms: %s", exc)
        return []


def _resolve_gpu_relax(use_gpu_relax: bool) -> bool:
    """Only request GPU relax when OpenMM exposes a CUDA platform."""
    if not use_gpu_relax:
        return False

    platform_names = _get_openmm_platform_names()
    if "CUDA" in platform_names:
        return True

    if platform_names:
        logging.warning(
            "OpenMM CUDA platform is unavailable; falling back to CPU relax. "
            "Available platforms: %s",
            ", ".join(platform_names),
        )
    else:
        logging.warning(
            "OpenMM CUDA platform is unavailable; falling back to CPU relax."
        )
    return False


def _on_gpu() -> bool:
    """True if jax has a GPU (CUDA or ROCm); Pallas and the fused CUDA kernels need one."""
    try:
        import jax
        return jax.devices()[0].platform == "gpu"
    except Exception:
        return False


def _compute_capability():
    """Give the GPU compute capability as an integer, e.g. 75, or None."""
    try:
        import jax
        cc = str(jax.devices()[0].compute_capability)
        return int(round(float(cc) * 10)) if "." in cc else int(cc)
    except Exception:
        return None


def _load_haiku_params(data_module, *, model_name: str, data_dir: str, use_fuse: bool):
    """Load Haiku params, keeping their layout consistent with ``use_fuse``.

    ``fuse_projection_weights=True`` in the model config expects the
    triangle-multiplication parameters to be pre-fused inside the loaded
    ``.npz``. If the installed ``alphafold.model.data.get_model_haiku_params``
    supports a ``fuse``/``use_fuse`` keyword, we pass it through so loading
    matches the config. If it doesn't support that keyword and fusion was
    requested, we fail fast here with a clear message instead of letting
    Haiku raise an opaque "Unable to retrieve parameter ..." deep inside the
    forward pass.
    """
    import inspect

    sig = inspect.signature(data_module.get_model_haiku_params)
    fuse_kwarg = None
    for candidate in ("fuse", "use_fuse"):
        if candidate in sig.parameters:
            fuse_kwarg = candidate
            break

    if fuse_kwarg is not None:
        return data_module.get_model_haiku_params(
            model_name=model_name, data_dir=data_dir, **{fuse_kwarg: use_fuse}
        )

    if use_fuse:
        raise ValueError(
            "use_fuse=True was requested, but this installation's "
            "alphafold.model.data.get_model_haiku_params(...) has no "
            "'fuse'/'use_fuse' keyword, so the loaded checkpoint's parameter "
            "layout cannot be guaranteed to match "
            "fuse_projection_weights=True. Either upgrade/patch that "
            "function to support fusing on load, or pass use_fuse=False."
        )
    return data_module.get_model_haiku_params(model_name=model_name, data_dir=data_dir)


def _warn_if_kernels_missing(cc) -> None:
    """Warn (rather than fail) when the legacy CUDA kernels aren't available for this GPU."""
    try:
        from alphafold.model import volta_attn
        if cc is not None and volta_attn.available(cc) and volta_attn.ops_available(cc):
            return
    except Exception:
        pass
    logging.warning("no colabfold-legacy-kernels for sm_%s; falling back to XLA", cc or "unknown")


def _jnp_to_np(output):
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


# Large outputs (N x N x 64 for most) that postprocess does not need; they stay in result_<model>.pkl.
_HEAVY_RESULT_KEYS = (
    "distogram", "masked_msa", "experimentally_resolved",
    "predicted_lddt", "structure_module", "aligned_confidence_probs",
)


def _drop_heavy_outputs(prediction_result: Dict) -> Dict:
    """Drops the large outputs from a result kept in memory until postprocess."""
    for key in _HEAVY_RESULT_KEYS:
        prediction_result.pop(key, None)
    return prediction_result


@contextlib.contextmanager
def _open_atomic(path: str, mode: str = "w"):
    """Write to a temporary file next to ``path``, renamed over it once complete.

    A job killed mid-write (systemd-oomd kills the whole process group) then never
    leaves a truncated pickle or PDB that a resumed run would trust.
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, mode) as f:
            yield f
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _dump_pickle(obj: Dict, path: str) -> None:
    """Protocol 5 writes NumPy arrays from their own memory; protocol 4 first copies
    every array into the pickler's memo, doubling the result in RAM."""
    with _open_atomic(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


def _calculate_bin_centers(breaks: np.ndarray) -> np.ndarray:
    """As alphafold.common.confidence: bin centres, plus a catch-all last bin."""
    step = breaks[1] - breaks[0]
    bin_centers = breaks + step / 2
    return np.concatenate([bin_centers, [bin_centers[-1] + step]], axis=0)


def _aligned_error_scores(
    *,
    breaks: np.ndarray,
    logits: np.ndarray = None,
    probs: np.ndarray = None,
    asym_id: np.ndarray = None,
    row_chunk_size: int = PAE_ROW_CHUNK_SIZE,
) -> Dict[str, Any]:
    """PAE matrix, pTM and (with ``asym_id``) ipTM from the PAE head, block by block.

    The formulas are those of alphafold.common.confidence
    (compute_predicted_aligned_error and predicted_tm_score with unit residue
    weights), but every quantity of a residue pair depends only on its row, so the
    rows are processed ``row_chunk_size`` at a time. The softmax of the whole
    N x N x bins logits, and the temporaries of the same size that scipy and NumPy
    create around it, are never held at once.

    Pass either the head's ``logits`` or its already normalised ``probs``.
    """
    source = logits if logits is not None else probs
    num_res = source.shape[0]
    bin_centers = _calculate_bin_centers(np.asarray(breaks, dtype=np.float64))
    if source.shape[-1] != bin_centers.size:
        raise ValueError(
            f"PAE head has {source.shape[-1]} bins but its breaks give {bin_centers.size}")
    # d_0(num_res) of the TM-score, eqn. (5) in Yang & Skolnick (2004).
    d0 = 1.24 * (max(num_res, 19) - 15) ** (1.0 / 3) - 1.8
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
    if asym_id is not None:
        asym_id = np.asarray(asym_id).reshape(-1)[:num_res]

    pae = np.empty((num_res, num_res), dtype=np.float32)
    ptm_per_row = np.empty(num_res)
    iptm_per_row = np.empty(num_res) if asym_id is not None else None
    for start in range(0, num_res, row_chunk_size):
        stop = min(start + row_chunk_size, num_res)
        block = np.array(source[start:stop], dtype=np.float32)
        if logits is not None:
            block -= block.max(axis=-1, keepdims=True)
            np.exp(block, out=block)
            block /= block.sum(axis=-1, keepdims=True)
        pae[start:stop] = block @ bin_centers
        tm_term = block @ tm_per_bin  # [rows, num_res]
        del block
        ptm_per_row[start:stop] = tm_term.sum(axis=-1) / (1e-8 + num_res)
        if asym_id is not None:
            pair_mask = asym_id[start:stop, None] != asym_id[None, :]
            iptm_per_row[start:stop] = (
                (tm_term * pair_mask).sum(axis=-1) / (1e-8 + pair_mask.sum(axis=-1)))
    scores = {
        "predicted_aligned_error": pae,
        "max_predicted_aligned_error": np.asarray(bin_centers[-1]),
        "ptm": np.asarray(ptm_per_row.max()),
    }
    if iptm_per_row is not None:
        scores["iptm"] = np.asarray(iptm_per_row.max())
    return scores


def _finalize_scores(
    result: Dict,
    *,
    multimer_mode: bool,
    total_num_res: int,
    asym_id: np.ndarray = None,
) -> bool:
    """Bring one prediction's scores to their final form, in place.

    - PAE still as the head's raw output (logits): compute the PAE matrix, pTM and
      ipTM from it with ``_aligned_error_scores``.
    - Padded prediction (``--desired_num_res``): pLDDT and the PAE arrays are cut to
      the real residues, and pTM/ipTM recomputed over them from
      ``aligned_confidence_probs``.
    - Multimer ranking on ColabFold's 0-100 scale: brought to 0-1.

    Idempotent. Returns whether anything changed, i.e. whether a pickle written
    before needs rewriting.
    """
    n = int(total_num_res)
    plddt = np.asarray(result["plddt"])
    padded = plddt.shape[0] > n
    pae_entry = result.get("predicted_aligned_error")
    scores = None
    if isinstance(pae_entry, Mapping):
        if asym_id is None:
            asym_id = pae_entry.get("asym_id")
        scores = _aligned_error_scores(
            logits=np.asarray(pae_entry["logits"])[:n, :n],
            breaks=np.asarray(pae_entry["breaks"]),
            asym_id=asym_id if multimer_mode else None,
        )
    elif padded and "aligned_confidence_probs" in result:
        probs = np.asarray(result["aligned_confidence_probs"])[:n, :n]
        num_bins = probs.shape[-1]
        max_pae = float(result["max_predicted_aligned_error"])
        # AlphaFold's bins: edges 0, s, 2s, ... and a catch-all bin, whose centre
        # (the maximum predicted error) is (num_bins - 0.5) * s.
        step = max_pae / (num_bins - 0.5)
        scores = _aligned_error_scores(
            probs=probs,
            breaks=np.arange(num_bins - 1) * step,
            asym_id=asym_id if multimer_mode else None,
        )
        result["aligned_confidence_probs"] = probs
    elif padded:
        logging.warning(
            "Padded prediction without aligned_confidence_probs: pLDDT and PAE are "
            "cut to the real residues, but pTM and ipTM keep the model's values.")
        if isinstance(pae_entry, np.ndarray):
            result["predicted_aligned_error"] = pae_entry[:n, :n]

    changed = scores is not None or padded
    if scores is not None:
        if multimer_mode and "iptm" not in scores:
            logging.warning("No chain IDs to recompute ipTM; keeping the model's value.")
        result.update(scores)
    if padded:
        result["plddt"] = plddt[:n]
    if not multimer_mode:
        if changed:
            # Monomer models rank by mean pLDDT.
            result["ranking_confidence"] = np.mean(np.asarray(result["plddt"]))
    elif scores is not None and "iptm" in scores:
        result["ranking_confidence"] = 0.8 * result["iptm"] + 0.2 * result["ptm"]
    elif result.get("ranking_confidence", 0) > 1:
        result["ranking_confidence"] = result["ranking_confidence"] / 100
        changed = True
    return changed


def _residue_b_factors(pdb_path: str) -> List[str]:
    """Each residue's B-factor: AlphaFold's pLDDT."""
    values = []
    last_residue = None
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                residue = line[21:27]  # chain ID, residue number and insertion code
                if residue != last_residue:
                    values.append(line[60:66].strip())
                    last_residue = residue
    return values


def _pdb_holds_prediction(pdb_path: str, prediction_result: Dict) -> bool:
    """Whether unrelaxed_<model>.pdb holds this pickle's model: its B-factors are the
    pickle's pLDDT. Only the real residues are compared, which come before any
    padding, and the pickle's pLDDT may already be cut to them."""
    b_factors = _residue_b_factors(pdb_path)
    plddt = np.asarray(prediction_result.get("plddt", ()))
    n = min(len(b_factors), len(plddt))
    return n > 0 and b_factors[:n] == [f"{value:.2f}" for value in plddt[:n]]


def _msa_depths(
    evoformer_config: Mapping,
    num_predictions: int,
    *,
    msa_depth=None,
    msa_depth_scan: bool = False,
) -> List[tuple]:
    """``(num_msa, num_extra_msa)`` for each prediction of one model.

    A fixed ``--msa_depth`` applies to every prediction, with the extra MSA at
    about four times the depth, as in the AlphaFold 2 configs. A scan spaces the
    depths logarithmically from 16 (32 for the extra MSA) up to the model's own
    defaults.
    """
    if msa_depth:
        num_msa = int(msa_depth)
        return [(num_msa, num_msa * 4)] * num_predictions
    msa_ranges = np.rint(
        np.logspace(np.log10(16), np.log10(evoformer_config["num_msa"]), num_predictions)
    ).astype(int)
    extra_msa_ranges = np.rint(
        np.logspace(np.log10(32), np.log10(evoformer_config["num_extra_msa"]), num_predictions)
    ).astype(int)
    return [(int(a), int(b)) for a, b in zip(msa_ranges, extra_msa_ranges)]


def _is_colabfold_multimer_runner(model_runner) -> bool:
    """Whether ``model_runner`` is alphafold-colabfold's RunModel for a multimer model,
    whose predict() recycles in a Python loop."""
    import inspect
    try:
        parameters = inspect.signature(model_runner.predict).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return ("callback" in parameters
            and hasattr(model_runner, "apply") and hasattr(model_runner, "init_params")
            and bool(getattr(model_runner, "multimer_mode", False)))


def _to_host_float16(tree):
    """ColabFold's conversion of the model outputs: every array to float16 NumPy."""
    if isinstance(tree, Mapping):
        return {k: _to_host_float16(v) for k, v in tree.items()}
    return np.asarray(tree, np.float16)


def _predict_keeping_recycles_on_gpu(model_runner, feat: Dict, random_seed: int) -> Dict:
    """alphafold-colabfold's RunModel.predict for a multimer model, with the recycling
    state kept on the GPU.

    ColabFold recycles in a Python loop and, at every recycle, copies all the outputs
    and the recycled pair representation (L x L x 128) to the host as float16, then the
    pair back to the GPU: hundreds of MB allocated and freed per recycle and per
    model. With several jobs in parallel that churn is the memory pressure systemd-oomd
    kills for. Here only the two scalars that decide early stopping leave the GPU at
    each recycle, and the outputs once, at the end. The steps, random keys, stopping
    rules and output types are ColabFold's; the recycled state stays float16, as
    ColabFold feeds it back, so the compiled network is the same.
    """
    import jax
    import jax.numpy as jnp

    model_runner.init_params(feat)
    model_config = model_runner.config.model
    num_res = feat["aatype"].shape[0]
    prev = {
        "prev_msa_first_row": jnp.zeros([num_res, 256], dtype=jnp.float16),
        "prev_pair": jnp.zeros([num_res, num_res, 128], dtype=jnp.float16),
    }
    if "all_atom_positions" in feat:
        prev["prev_pos"] = jnp.asarray(np.asarray(feat["all_atom_positions"], np.float16))
    else:
        prev["prev_pos"] = jnp.zeros([num_res, 37, 3], dtype=jnp.float16)

    key = jax.random.PRNGKey(random_seed)
    for r in range(model_config.num_recycle + 1):
        key, sub_key = jax.random.split(key)
        # Free the previous recycle's outputs before computing the next ones.
        result = None
        result = model_runner.apply(model_runner.params, sub_key, {**feat, "prev": prev})
        prev = {k: v.astype(jnp.float16) for k, v in result.pop("prev").items()}
        if float(result["ranking_confidence"]) > model_config.stop_at_score:
            break
        if r > 0 and float(result["tol"]) < model_config.recycle_early_stop_tolerance:
            break
    del prev
    return _to_host_float16(result)


def _normalize_prediction_result(
    prediction_result: Any, *, model_name: str
) -> Dict[str, Any]:
    """Accept the supported AF2 runner return shapes and return a mutable dict."""
    if isinstance(prediction_result, Mapping):
        return dict(prediction_result)

    if isinstance(prediction_result, tuple):
        if len(prediction_result) == 2 and isinstance(prediction_result[0], Mapping):
            auxiliary_output = prediction_result[1]
            #logging.warning(
            #    "Model %s returned a (prediction_result, auxiliary_output) tuple; "
            #    "ignoring auxiliary output of type %s.",
            #    model_name,
            #    type(auxiliary_output).__name__,
            #)
            return dict(prediction_result[0])

        first_type = type(prediction_result[0]).__name__ if prediction_result else "n/a"
        raise TypeError(
            "model_runner.predict must return a mapping or a (mapping, auxiliary) "
            f"tuple; got tuple of length {len(prediction_result)} with first "
            f"element type {first_type}."
        )

    raise TypeError(
        "model_runner.predict must return a mapping or a (mapping, auxiliary) "
        f"tuple; got {type(prediction_result).__name__}."
    )


def _save_pae_json_file(pae: np.ndarray, max_pae: float, output_dir: str, model_name: str) -> None:
    """
    Check prediction result for PAE data and save to a JSON file if present.

    Args:
    pae: The n_res x n_res PAE array.
    max_pae: The maximum possible PAE value.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
    """
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f'PAE must be a square matrix, got {pae.shape}')
    # Same output as generate_pae_json, written row by row rather than built as one
    # N x N list of Python floats and one string.
    pae_json_output_path = os.path.join(output_dir, f'pae_{model_name}.json')
    with _open_atomic(pae_json_output_path) as f:
        f.write('[{"predicted_aligned_error":[')
        for i, row in enumerate(pae):
            if i:
                f.write(',')
            f.write(json.dumps(np.round(row.astype(np.float64), decimals=1).tolist(),
                               separators=(',', ':')))
        f.write('],"max_predicted_aligned_error":')
        f.write(json.dumps(max_pae))
        f.write('}]')


def _save_confidence_json_file(plddt: np.ndarray, output_dir: str, model_name: str) -> None:
    """
    Check prediction result for confidence data and save to a JSON file if present.
    Args:
        plddt: The n_res x 1 pLDDT array.
        output_dir: Directory to which files are saved.
        model_name: Name of a model.
    """
    confidence_json = generate_confidence_json(plddt)
    confidence_json_output_path = os.path.join(
        output_dir, f'confidence_{model_name}.json')
    with open(confidence_json_output_path, 'w') as f:
        f.write(confidence_json)


def _read_from_json_if_exists(json_path: str) -> Dict:
    """Reads a JSON file or creates it with default data if it does not exist."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data


def _reset_template_features(feature_dict: Dict) -> None:
    """
    Resets specific features within a dictionary to their default state.
    - 'template_aatype' and 'template_all_atom_positions' are reset to zeros.
    - 'template_all_atom_masks' is reset to ones.
    - 'num_templates' is set to one.
    Parameters:
    feature_dict (Dict[str, np.ndarray]): The feature dictionary to be modified.
    """
    seq_length = feature_dict["seq_length"]
    for key, value in feature_dict.items():
        if key == "template_aatype":
            feature_dict[key] = np.zeros((1, seq_length), dtype='int64')
        elif key == "template_all_atom_positions":
            feature_dict[key] = np.zeros(
                (1, seq_length, 37, 3), dtype='float32')
        elif key == "template_all_atom_mask":
            feature_dict[key] = np.ones((1, seq_length, 37), dtype='float32')
        elif key == "num_templates":
            feature_dict[key] = np.ones_like(value)


def _normalize_asym_id(feature_dict: Dict, fallback_feature_dict: Dict = None) -> Dict:
    """Returns a feature dictionary with deterministic 0-based chain indices.

    AlphaFold multimer features may use arbitrary positive ``asym_id`` values, and
    these values can change across model seeds. ``protein.to_pdb`` interprets those
    integers directly as chain indices, so we remap unique IDs to ``0..N-1`` in
    first-appearance order to keep chain IDs stable (A, B, C, ...).

    ``pad_input_features`` zero-pads ``asym_id`` to the batch-wide residue bound.
    Those trailing zeros are padding rather than a chain, so they rank after every
    real chain instead of displacing them.
    """
    asym_id = feature_dict.get("asym_id")
    if asym_id is None and fallback_feature_dict is not None:
        asym_id = fallback_feature_dict.get("asym_id")
    if asym_id is None:
        return feature_dict

    asym_id = np.asarray(asym_id).reshape(-1)

    # AlphaFold numbers multimer chains from 1, so a genuinely 0-based encoding
    # starts at 0 rather than ending there.
    padding_mask = np.zeros(asym_id.shape, dtype=bool)
    if asym_id.size and asym_id[0] != 0:
        non_padding = np.flatnonzero(asym_id != 0)
        padding_mask[(non_padding[-1] + 1) if non_padding.size else 0:] = True

    # Deterministic contiguous remapping in first-appearance order.
    unique_ids, first_indices, inverse_indices = np.unique(
        asym_id[~padding_mask], return_index=True, return_inverse=True
    )
    appearance_order = np.argsort(first_indices)
    ranks = np.empty_like(appearance_order)
    ranks[appearance_order] = np.arange(appearance_order.size)

    remapped_indices = np.empty(asym_id.shape, dtype=np.int32)
    remapped_indices[~padding_mask] = ranks[inverse_indices.reshape(-1)]
    remapped_indices[padding_mask] = unique_ids.size

    normalized_feature_dict = dict(feature_dict)
    normalized_feature_dict["asym_id"] = remapped_indices
    return normalized_feature_dict


def _normalise_num_alignments_for_debug(feature_dict: Dict[str, Any]) -> int:
    msa_rows = feature_dict.get("msa")
    if msa_rows is None:
        return 0
    max_rows = int(np.asarray(msa_rows).shape[0])
    num_alignments = feature_dict.get("num_alignments")
    if num_alignments is None:
        return max_rows
    return max(0, min(int(np.asarray(num_alignments).reshape(-1)[0]), max_rows))


def _query_sequence_for_debug(
    multimeric_object: Union[MultimericObject, MonomericObject, ChoppedObject]
) -> str:
    if isinstance(multimeric_object, MultimericObject):
        return "".join(multimeric_object.input_seqs)
    return multimeric_object.sequence


def _write_processed_msa_debug_artifact(
    *,
    processed_feature_dict: Dict[str, Any],
    multimeric_object: Union[MultimericObject, MonomericObject, ChoppedObject],
    output_dir: str,
    model_name: str,
) -> None:
    msa_rows = processed_feature_dict.get("msa")
    if msa_rows is None:
        return

    num_alignments = _normalise_num_alignments_for_debug(processed_feature_dict)
    if num_alignments <= 0:
        return

    deletion_rows = processed_feature_dict.get("deletion_matrix_int")
    if deletion_rows is None:
        deletion_rows = processed_feature_dict.get("deletion_matrix")

    a3m_text = msa_rows_and_deletions_to_a3m(
        msa_rows=np.asarray(msa_rows)[:num_alignments],
        deletion_rows=(
            None if deletion_rows is None else np.asarray(deletion_rows)[:num_alignments]
        ),
        query_sequence=_query_sequence_for_debug(multimeric_object),
    )
    out_path = os.path.join(output_dir, f"{model_name}_processed_msa.a3m")
    with open(out_path, "wt") as handle:
        handle.write(a3m_text)
    logging.info(f"Wrote processed MSA A3M to {out_path}")


def _decode_debug_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _sanitize_debug_filename(value: Any, fallback: str) -> str:
    text = _decode_debug_value(value).strip()
    if not text:
        text = fallback
    text = text.replace(os.sep, "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _template_aatype_to_indices(template_aatype: np.ndarray) -> np.ndarray:
    template_aatype = np.asarray(template_aatype)
    if template_aatype.ndim == 2:
        hh_ids = np.argmax(template_aatype, axis=-1)
        aatype = np.array(
            [
                residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[int(index)]
                for index in hh_ids
            ],
            dtype=np.int32,
        )
        aatype[aatype == 21] = 20
        return aatype

    if template_aatype.ndim == 1:
        aatype = template_aatype.astype(np.int32, copy=False)
        if np.any(aatype > residue_constants.restype_num):
            mapped = np.array(
                [
                    residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[int(index)]
                    for index in aatype
                ],
                dtype=np.int32,
            )
            mapped[mapped == 21] = 20
            return mapped
        return np.clip(aatype, 0, residue_constants.restype_num)

    raise ValueError(
        f"Unsupported template_aatype rank for debug dump: {template_aatype.shape}"
    )


def _write_processed_template_debug_artifacts(
    *,
    processed_feature_dict: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    template_positions = processed_feature_dict.get("template_all_atom_positions")
    if template_positions is None:
        return

    template_masks = processed_feature_dict.get("template_all_atom_masks")
    if template_masks is None:
        template_masks = processed_feature_dict.get("template_all_atom_mask")
    if template_masks is None:
        return

    template_aatype = processed_feature_dict.get("template_aatype")
    if template_aatype is None:
        return

    residue_index = np.asarray(
        processed_feature_dict.get(
            "residue_index",
            np.arange(np.asarray(template_positions).shape[1], dtype=np.int32),
        )
    ).reshape(-1)
    asym_id = processed_feature_dict.get("asym_id")
    if asym_id is None:
        chain_index = np.zeros_like(residue_index, dtype=np.int32)
    else:
        chain_index = np.asarray(_normalize_asym_id(processed_feature_dict)["asym_id"]).reshape(-1)

    debug_dir = os.path.join(output_dir, "templates_debug")
    os.makedirs(debug_dir, exist_ok=True)

    domain_names = processed_feature_dict.get("template_domain_names")
    if domain_names is None:
        domain_names = [f"template_{i}" for i in range(np.asarray(template_positions).shape[0])]

    for template_index, (positions, mask, aa, domain_name) in enumerate(
        zip(template_positions, template_masks, template_aatype, domain_names, strict=True)
    ):
        if not np.any(mask):
            continue
        try:
            template_protein = protein.Protein(
                atom_positions=np.asarray(positions),
                atom_mask=np.asarray(mask),
                aatype=_template_aatype_to_indices(np.asarray(aa)),
                residue_index=residue_index,
                chain_index=chain_index,
                b_factors=np.zeros_like(np.asarray(mask), dtype=float),
            )
            template_name = _sanitize_debug_filename(
                domain_name, f"template_{template_index}"
            )
            debug_path = os.path.join(
                debug_dir, f"{model_name}_{template_name}_idx{template_index}.pdb"
            )
            with open(debug_path, "wt") as handle:
                handle.write(protein.to_pdb(template_protein))
            logging.info(f"Wrote processed template PDB to {debug_path}")
        except Exception as exc:
            error_path = os.path.join(
                debug_dir, f"ERROR_{model_name}_template_{template_index}.txt"
            )
            with open(error_path, "wt") as handle:
                handle.write(f"Error: {exc}\n")
                handle.write(f"Model: {model_name}\n")
                handle.write(f"Template index: {template_index}\n")
                handle.write(
                    f"Domain name: {_decode_debug_value(domain_name) if domain_name is not None else ''}\n"
                )
            logging.error(
                f"Failed to write processed template debug artifact for {model_name} "
                f"template {template_index}: {exc}"
            )

def _confidence_category(score: float) -> str:
  """Categorizes pLDDT into: disordered (D), low (L), medium (M), high (H)."""
  if 0 <= score < 50:
    return 'D'
  if 50 <= score < 70:
    return 'L'
  elif 70 <= score < 90:
    return 'M'
  elif 90 <= score <= 100:
    return 'H'
  else:
    raise ValueError(f'Invalid pLDDT score {score}')

def generate_pae_json(pae: np.ndarray, max_pae: float) -> str:
  """Returns the PAE in the same format as is used in the AFDB.

  Note that the values are presented as floats to 1 decimal place, whereas AFDB
  returns integer values.

  Args:
    pae: The n_res x n_res PAE array.
    max_pae: The maximum possible PAE value.

  Returns:
    PAE output format as a JSON string.
  """
  # Check the PAE array is the correct shape.
  if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
    raise ValueError(f'PAE must be a square matrix, got {pae.shape}')

  # Round the predicted aligned errors to 1 decimal place.
  rounded_errors = np.round(pae.astype(np.float64), decimals=1)
  formatted_output = [{
      'predicted_aligned_error': rounded_errors.tolist(),
      'max_predicted_aligned_error': max_pae,
  }]
  return json.dumps(formatted_output, indent=None, separators=(',', ':'))

def generate_confidence_json(plddt: np.ndarray) -> str:
  """Returns JSON with confidence score and category for every residue.

  Args:
    plddt: Per-residue confidence metric data.

  Returns:
    String with a formatted JSON.

  Raises:
    ValueError: If `plddt` has a rank different than 1.
  """
  if plddt.ndim != 1:
    raise ValueError(f'The plddt array must be rank 1, got: {plddt.shape}.')

  confidence = {
      'residueNumber': list(range(1, len(plddt) + 1)),
      'confidenceScore': [round(float(s), 2) for s in plddt],
      'confidenceCategory': [_confidence_category(s) for s in plddt],
  }
  return json.dumps(confidence, indent=None, separators=(',', ':'))

class AlphaFold2Backend(FoldingBackend):
    """
    A backend to perform structure prediction using AlphaFold.
    """

    @staticmethod
    def setup(
        model_name: str,
        num_cycle: int,
        model_dir: str,
        num_predictions_per_model: int,
        msa_depth_scan=False,
        model_names_custom: List[str] = None,
        msa_depth=None,
        dropout=False,
        use_fast_kernels: bool = True,
        kernel_backend: str = "auto",
        use_fuse: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures multiple AlphaFold model runners.

        Parameters
        ----------
        model_name : str
            The preset model configuration name.
        num_cycle : int
            The number of recycling iterations to be used in prediction.
        model_dir : str
            The directory containing model parameters.
        num_predictions_per_model : int
            The number of multimer predictions to perform for each model.
        msa_depth_scan : bool, optional
            Whether to adjust MSA depth logarithmically, default is False.
        model_names_custom : list, optional
            A list of strings that specify which models to run, default is None, meaning all 5 models will be used
        msa_depth : int or None, optional
            A specific MSA depth to use, default is None.
        allow_resume : bool, optional
            If set to True, resumes prediction from partially completed runs, default is True.
        dropout : bool, optional
            If set to True, use dropout when inferring for more diverse predictions, default is False.
        use_fast_kernels : bool, optional
            If set to True, enables the fused attention kernels (Pallas/Triton on
            sm_80+, or the legacy CUDA kernels on older GPUs) instead of plain XLA.
            Default is True; set to False to force plain XLA (e.g. while
            diagnosing a kernel-shape error such as a Pallas/Triton
            "all dimensions of b must be >= 16").
        kernel_backend : str, optional
            Which fused-kernel backend to use when ``use_fast_kernels`` is True:
            "auto" (default, picks "pallas" on sm_80+ and "cuda_legacy" otherwise),
            "pallas", or "cuda_legacy". Has no effect if ``use_fast_kernels`` is False.
        use_fuse : bool, optional
            If set to True (default), enables fused triangle-multiplication
            projections (``fuse_projection_weights``) in the evoformer and
            template pair stacks, and the weight loader is asked to match
            (see below). Set to False if you hit "Unable to retrieve
            parameter ..." for a triangle_multiplication module, which means
            your ``.npz`` checkpoint is not in fused layout.
        **kwargs : dict
            Additional keyword arguments for model runner configuration.

        Returns
        -------
        Dict
            A dictionary containing the configured model runners, and other settings

        Raises
        ------
        Exception
            If provided custom model names are not part of the available models.
        ValueError
            If ``kernel_backend="pallas"`` is requested explicitly on a GPU older
            than sm_80.
        """

        _ensure_typing_dataclass_transform()
        from alphafold.model import config
        from alphafold.model import data, model

        # AlphaFold2 inference is JAX-compiled, and a fresh process recompiles every
        # model runner. Honour a persistent on-disk compile cache when one is given.
        enable_persistent_compilation_cache(os.environ.get("JAX_COMPILATION_CACHE_DIR"))
        

        if kernel_backend != "auto" and not use_fast_kernels:
            logging.warning("kernel_backend has no effect without use_fast_kernels")

        if use_fast_kernels and not _on_gpu():
            logging.warning("fused kernels need a GPU; ignoring use_fast_kernels")
            use_fast_kernels = False

        num_ensemble = 1
        model_runners = {}
        model_names = config.MODEL_PRESETS[model_name]
        # add model names of older versionsto be compatible with older version of AlphaFold Multimer
        old_model_names = (
          'model_1_multimer',
          'model_2_multimer',
          'model_3_multimer',
          'model_4_multimer',
          'model_5_multimer',
          'model_1_multimer_v2',
          'model_2_multimer_v2',
          'model_3_multimer_v2',
          'model_4_multimer_v2',
          'model_5_multimer_v2',
      )
        if model_names_custom:
            model_names_custom = tuple(model_names_custom)
            if all(x in model_names + old_model_names for x in model_names_custom):
                model_names = model_names_custom
            else:
                raise Exception(
                    f"Provided model names {model_names_custom} not part of available {model_names + old_model_names}"
                )

        cc = _compute_capability()
        backend = kernel_backend
        if backend == "auto":
            # XLA gates Pallas/Triton to sm_80+.
            backend = "cuda_legacy" if cc is not None and cc < 80 else "pallas"
        if use_fast_kernels and backend == "pallas" and cc is not None and cc < 80:
            raise ValueError(
                f"--kernel-backend pallas needs sm_80+, this GPU is sm_{cc}; "
                "use auto or cuda_legacy")
        if use_fast_kernels and backend == "cuda_legacy":
            _warn_if_kernels_missing(cc)

        def configured_model(name: str, num_msa=None, num_extra_msa=None):
            """A fresh model config for one runner.

            ``RunModel`` keeps a reference to its config and traces the network from
            it on first use, so every runner needs its own copy: rewriting one shared
            config per MSA depth left every runner at the last depth.
            """
            model_config = config.model_config(name)
            model_config.model.num_ensemble_eval = num_ensemble
            model_config["model"].update({"num_recycle": num_cycle})
            if dropout:
                model_config.model.global_config.eval_dropout = True
            if num_msa is not None:
                model_config["model"]["embeddings_and_evoformer"].update(
                    {"num_msa": int(num_msa), "num_extra_msa": int(num_extra_msa)})
            model_config.model.global_config.use_pallas = use_fast_kernels
            model_config.model.global_config.kernel_backend = backend
            model_config.model.global_config.compute_capability = cc
            # Volta/Turing tensor cores have no bfloat16
            if use_fast_kernels and backend == "cuda_legacy":
                model_config.model.global_config.half_dtype = "float16"
            # Fused triangle-multiplication projections. Only turn this on if the
            # checkpoint you are loading was actually packed in fused format --
            # see the use_fuse docstring above for why a mismatch here raises a
            # Haiku "Unable to retrieve parameter" error instead of a silent
            # slowdown.
            evoformer = model_config.model.embeddings_and_evoformer
            evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
            evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
            if "multimer" in name:
                evoformer.template.template_pair_stack.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
                evoformer.template.template_pair_stack.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
            return model_config

        for model_name in model_names:
            # The parameters are large: load them once per model name and share them
            # between the runners built from it.
            model_params = _load_haiku_params(
                data, model_name=model_name, data_dir=model_dir, use_fuse=use_fuse
            )
            if not (msa_depth or msa_depth_scan):
                model_runner = model.RunModel(configured_model(model_name), model_params)
                for i in range(num_predictions_per_model):
                    model_runners[f"{model_name}_pred_{i}"] = model_runner
                continue

            depths = _msa_depths(
                configured_model(model_name)["model"]["embeddings_and_evoformer"],
                num_predictions_per_model,
                msa_depth=msa_depth,
                msa_depth_scan=msa_depth_scan,
            )
            runners_by_depth = {}
            for i, depth in enumerate(depths):
                if depth not in runners_by_depth:
                    runners_by_depth[depth] = model.RunModel(
                        configured_model(model_name, *depth), model_params)
                model_runners[f"{model_name}_pred_{i}_msa_{depth[0]}"] = runners_by_depth[depth]

        return {"model_runners": model_runners}

    @staticmethod
    def predict_individual_job(
        model_runners: Dict,
        multimeric_object: Union[MultimericObject, MonomericObject],
        allow_resume: bool,
        skip_templates: bool,
        output_dir: Dict,
        random_seed: int,
        **kwargs,
    ) -> dict:
        """
        Executes structure predictions using configured AlphaFold models on one individual job.

        Parameters
        ----------
        model_runners : Dict
            Configured model runners with model names as keys.
        allow_resume : bool
            If set to True, resumes prediction from partially completed runs.
        skip_templates : bool
            Do not use templates for prediction.
        multimeric_object : MultimericObject
            An object containing features of the multimeric proteins or monomeric protein,
            for the sake of simplicity, it is named as multimeric_object but can be a MonomericObject.
        output_dir : str
            The directory to save prediction results and PDB files.
        random_seed : int, optional
            A seed for random number generation to ensure reproducibility, default is 42.
        **kwargs : dict
            Additional keyword arguments for prediction.

        Returns
        -------
        Dict
            A dictionary mapping model names with corresponding prediction results and predicted proteins.

        Raises
        ------
        ValueError
            If multimeric mode is enabled but no valid templates are found.
        ValueError
            If multimeric mode and skip templates are enabled at the same time
        """
        timings = {}
        prediction_results = {}
        START = 0
        multimeric_mode = multimeric_object.multimeric_mode if hasattr(multimeric_object, "multimeric_mode") else None
        multimer_mode = type(multimeric_object) == MultimericObject
        seqs = multimeric_object.input_seqs if hasattr(multimeric_object, "input_seqs") else [multimeric_object.sequence]
        total_num_res = sum(len(s) for s in seqs)
        original_feature_dict = copy(multimeric_object.feature_dict)
        if allow_resume:
            logging.info(
            f"Now running predictions on {multimeric_object.description}. Checking existing results...")
            for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
                unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
                result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
                # If --remove_result_pickle=True the predictions are done anew.
                if not (os.path.exists(unrelaxed_pdb_path) and os.path.exists(result_output_path)):
                    break
                with open(result_output_path, "rb") as f:
                    prediction_result = pickle.load(f)
                if not _pdb_holds_prediction(unrelaxed_pdb_path, prediction_result):
                    # Killed between writing a new pickle and its PDB: the PDB is still
                    # the earlier prediction's, so the model is predicted again.
                    logging.info(
                        f"{unrelaxed_pdb_path} is not the model in {result_output_path}; "
                        f"predicting {model_name} again.")
                    break
                # A pickle from an older run may still need its final scores.
                rewrite = _finalize_scores(
                    prediction_result, multimer_mode=multimer_mode,
                    total_num_res=total_num_res,
                    asym_id=original_feature_dict.get("asym_id"))
                if "seqs" not in prediction_result:
                    prediction_result["seqs"] = seqs
                    rewrite = True
                if rewrite:
                    _dump_pickle(_jnp_to_np(prediction_result), result_output_path)
                # Read the PDB rather than process the features again to rebuild it.
                with open(unrelaxed_pdb_path, "r") as pdb:
                    prediction_result["unrelaxed_protein"] = protein.from_pdb_string(pdb.read())
                prediction_results[model_name] = _drop_heavy_outputs(prediction_result)
                del prediction_result
                START = model_index + 1
            gc.collect()
        logging.info(f"{START} models are already completed.")
        if START == len(model_runners):
            logging.info(
                f"All predictions for {multimeric_object.description} are already completed.")
            return prediction_results
        # first check whether the desired num_res and num_msa are specified for padding
        desired_num_res, desired_num_msa = kwargs.get(
            "desired_num_res", None), kwargs.get("desired_num_msa", None)
        if (desired_num_res is not None and
                type(multimeric_object) == MultimericObject):
            # Padding every fold in a batch to one shape lets a resident session compile
            # once instead of once per fold. num_res is the dimension that varies between
            # folds; the MSA dimension is fixed by the model config during
            # process_features, so default it to this fold's own depth (a no-op pad) when
            # a batch-wide bound was not supplied.
            if desired_num_msa is None:
                desired_num_msa = int(original_feature_dict['msa'].shape[0])
            pad_input_features(feature_dict=original_feature_dict,
                               desired_num_msa=desired_num_msa, desired_num_res=desired_num_res)
            original_feature_dict['num_alignments'] = np.array([desired_num_msa])
        num_models = len(model_runners)
        # Predict
        for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
            # Skip if results.pkl und unrelaxed.pdb are exist
            if model_index < START: continue
            t_0 = time.time()
            model_random_seed = model_index + random_seed * num_models
            processed_feature_dict = model_runner.process_features(
                original_feature_dict, random_seed=model_random_seed
            )
            # TODO: re-predict models if --allow_resume and previous predictions were done with templates
            if skip_templates:
                _reset_template_features(processed_feature_dict)
            timings[f"process_features_{model_name}"] = time.time() - t_0
            # Die if --multimeric_mode=True but no non-zero templates are in the feature dict
            if multimeric_mode:
                if "template_all_atom_positions" in processed_feature_dict:
                    if not np.any(
                        processed_feature_dict["template_all_atom_positions"]
                    ):
                        if skip_templates:
                            raise ValueError(
                                "You cannot skip templates in multimeric mode.")
                        raise ValueError(
                            "No valid templates found: all positions are zero."
                        )
                else:
                    raise ValueError(
                        "No template_all_atom_positions key found in processed_feature_dict."
                    )
            if kwargs.get("debug_msas", False):
                _write_processed_msa_debug_artifact(
                    processed_feature_dict=processed_feature_dict,
                    multimeric_object=multimeric_object,
                    output_dir=output_dir,
                    model_name=model_name,
                )
            if kwargs.get("debug_templates", False):
                _write_processed_template_debug_artifacts(
                    processed_feature_dict=processed_feature_dict,
                    output_dir=output_dir,
                    model_name=model_name,
                )
            t_0 = time.time()
            logging.info(
                f"Now running predictions on {multimeric_object.description} using {model_name}")
            if _is_colabfold_multimer_runner(model_runner):
                prediction_result = _predict_keeping_recycles_on_gpu(
                    model_runner, processed_feature_dict, model_random_seed)
            else:
                prediction_result = model_runner.predict(
                    processed_feature_dict, random_seed=model_random_seed
                )
            prediction_result = _normalize_prediction_result(
                prediction_result, model_name=model_name
            )
            t_diff = time.time() - t_0
            timings[f"predict_and_compile_{model_name}"] = t_diff
            logging.info(f"prediction costs : {t_diff} s")

            # Update prediction_result with input seqs and unrelaxed protein
            plddt_b_factors = np.repeat(
                prediction_result['plddt'][:,
                                           None], residue_constants.atom_type_num, axis=-1
            )
            
            normalized_feature_dict = _normalize_asym_id(
                processed_feature_dict, fallback_feature_dict=original_feature_dict
            )

            unrelaxed_protein = protein.from_prediction(
                features=normalized_feature_dict,
                result=prediction_result,
                b_factors=plddt_b_factors,
                remove_leading_feature_dimension=not model_runner.multimer_mode,
            )
            # Convert to NumPy and drop the references to the JAX arrays
            prediction_result = _jnp_to_np(prediction_result)
            # Final scores now, while this model's full output is in memory: postprocess
            # then never has to reload and rewrite a pickle.
            _finalize_scores(
                prediction_result, multimer_mode=multimer_mode,
                total_num_res=total_num_res,
                asym_id=original_feature_dict.get("asym_id"))
            prediction_result["seqs"] = seqs
            # Save prediction results to pickle file (before adding the protein), then
            # the PDB: a resumed run checks that the two match.
            result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
            _dump_pickle(prediction_result, result_output_path)
            unrelaxed_pdb_path = os.path.join(
                output_dir, f"unrelaxed_{model_name}.pdb")
            with _open_atomic(unrelaxed_pdb_path) as f:
                f.write(protein.to_pdb(unrelaxed_protein))
            prediction_result["unrelaxed_protein"] = unrelaxed_protein
            prediction_results[model_name] = _drop_heavy_outputs(prediction_result)
            # Save timings to json file
            timings_output_path = os.path.join(output_dir, "timings.json")
            with _open_atomic(timings_output_path) as f:
                f.write(json.dumps(timings, indent=4))
            # Free this model's features and outputs before predicting the next one:
            # arrays caught in reference cycles otherwise wait for a full collection.
            del processed_feature_dict, normalized_feature_dict, prediction_result, plddt_b_factors
            gc.collect()
        return prediction_results

    @staticmethod
    def predict(model_runners: Dict,
                objects_to_model: list,  # List of dicts with 'object' and 'output_dir'
                allow_resume: bool,
                skip_templates: bool,
                random_seed: int = 42,
                **kwargs):
        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']
            
            prediction_results = AlphaFold2Backend.predict_individual_job(
                model_runners=model_runners,
                multimeric_object=object_to_model,
                allow_resume=allow_resume,
                skip_templates=skip_templates,
                output_dir=output_dir,
                random_seed=random_seed,
                **kwargs
            )
            
            yield {'object': object_to_model, 
                   'prediction_results': prediction_results,
                   'output_dir': output_dir}
            
    @staticmethod
    def recalculate_confidence(prediction_results: Dict, multimer_mode: bool,
                               total_num_res: int, asym_id=None) -> Dict[str, Any]:
        """The entries of one prediction that its final scores change.

        See ``_finalize_scores``: PAE, pTM and ipTM from the PAE logits when the model
        returned them, cut to the real residues when the prediction was padded, and
        the multimer ranking brought from ColabFold's 0-100 scale to 0-1. Empty when
        the scores are already final.
        """
        updated = dict(prediction_results)
        _finalize_scores(updated, multimer_mode=multimer_mode,
                         total_num_res=total_num_res, asym_id=asym_id)
        return {k: v for k, v in updated.items()
                if k not in prediction_results or v is not prediction_results[k]}

    @staticmethod
    def postprocess(
        prediction_results: Dict,
        multimeric_object: MultimericObject,
        output_dir: str,
        features_directory: str,
        models_to_relax: ModelsToRelax,
        relax_best_score_threshold = None,
        compress_pickles: bool = False,
        remove_pickles: bool = False,
        remove_keys_from_pickles: bool = False,
        storage_mode: str = "vanilla",
        convert_to_modelcif: bool = False,
        use_gpu_relax: bool = True,
        pae_plot_style: str = "red_blue",

        **kwargs: Dict,
    ) -> None:
        """
        Performs post-processing operations on predicted protein structures and
        writes results and plots to output_dir.

        Parameters
        ----------
        prediction_results: Dict
            A dictionary mapping model names with corresponding prediction results.
        multimeric_object : MultimericObject
            The multimeric object containing the predicted structures and
            associated data.
        output_dir : str
            The directory where post-processed files and plots will be saved.
        features_directory: str
            The directory containing the features used for prediction. Used by the convert_to_modelcif script.
        models_to_relax : object
            Specifies which models' predictions to relax, defaults to ModelsToRelax enum.
        compress_pickles : bool, optional
            If True, zips the pickle files containing prediction results.
            Default is False.
        remove_pickles : bool, optional
            If True, removes the pickle files after post-processing is complete.
            Default is False.
        convert_to_modelcif : bool, optional
            If set to True, converts all predicted models to ModelCIF format, default is True.
        use_gpu_relax : bool, optional
            If set to True, utilizes GPU acceleration for the relaxation step, default is True.
        pae_plot_style : str, optional
            The style of the PAE plot, red and blue or AF database style, default is "red_blue".
        **kwargs : dict
            Additional keyword arguments for future extensions or custom
            post-processing steps.
        """
        relaxed_pdbs = {}
        ranking_confidences = {}
        iptm_scores = {}
        ptm_scores = {}
        multimer_mode = type(multimeric_object) == MultimericObject
        # Read timings.json if exists
        timings_path = os.path.join(output_dir, 'timings.json')
        timings = _read_from_json_if_exists(timings_path)
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        relax_metrics = _read_from_json_if_exists(relax_metrics_path)

        ranking_path = os.path.join(output_dir, "ranking_debug.json")
        multimeric_mode = multimeric_object.multimeric_mode if hasattr(multimeric_object, "multimeric_mode") else None

        total_num_res = sum([len(s) for s in multimeric_object.input_seqs]) if multimer_mode else len(multimeric_object.sequence)
        # Save plddt json files.
        for model_name, prediction_result in prediction_results.items():
            # The scores were finalised, and the pickle written, when the model was
            # predicted or resumed. Only results that did not go through
            # predict_individual_job still need it, and their pickle rewriting.
            updates = AlphaFold2Backend.recalculate_confidence(
                prediction_result, multimer_mode, total_num_res,
                asym_id=getattr(multimeric_object, "feature_dict", {}).get("asym_id"))
            if updates:
                prediction_result.update(updates)
                result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
                full_result = {}
                if os.path.exists(result_output_path):
                    with open(result_output_path, "rb") as f:
                        full_result = pickle.load(f)
                full_result.update(
                    {k: v for k, v in prediction_result.items() if k != "unrelaxed_protein"})
                _dump_pickle(_jnp_to_np(full_result), result_output_path)
                del full_result
            if 'iptm' in prediction_result:
                label = 'iptm+ptm'
                iptm_scores[model_name] = float(prediction_result['iptm'])
                cmplx = True
            else:
                label = 'plddts'
                ptm_scores[model_name] = float(prediction_result['ptm'])
                cmplx = False
            plddt = prediction_result['plddt']
            _save_confidence_json_file(plddt, output_dir, model_name)
            ranking_confidences[model_name] = float(prediction_result['ranking_confidence'])
            # Save and plot PAE if predicting multimer.
            if (
                    'predicted_aligned_error' in prediction_result
                    and 'max_predicted_aligned_error' in prediction_result
            ):
                pae = prediction_result['predicted_aligned_error']

                max_pae = prediction_result['max_predicted_aligned_error']
                _save_pae_json_file(pae, float(max_pae),output_dir, model_name)
        # Rank by model confidence.
        ranked_order = [
            model_name for model_name, confidence in
            sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

        # Save pae plots as *.png files.
        for idx, model_name in enumerate(ranked_order):
            prediction_result = prediction_results[model_name]
            figure_name = os.path.join(
                output_dir, f"pae_plot_ranked_{idx}.png")
            plot_pae_from_matrix(
                seqs=prediction_result['seqs'],
                pae_matrix=prediction_result['predicted_aligned_error'],
                figure_name=figure_name,
                ranking=idx
            )
            gc.collect()

        # Save ranking_debug.json
        with open(ranking_path, 'w') as f:
            json.dump(
                {label: ranking_confidences, 'order': ranked_order,
                 "iptm" if cmplx else "ptm": iptm_scores if cmplx else ptm_scores},
                f,
                indent=4
            )

        # Relax.
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS,
            exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=_resolve_gpu_relax(use_gpu_relax))

        to_relax = _select_models_to_relax(
            ranked_order,
            models_to_relax=models_to_relax,
            iptm_scores=iptm_scores,
            ptm_scores=ptm_scores,
            relax_best_score_threshold=relax_best_score_threshold,
        )

        for model_name in to_relax:
            if f'relax_{model_name}' in timings:
                continue
            t_0 = time.time()
            if 'unrelaxed_protein' in prediction_results[model_name].keys():
                unrelaxed_protein = prediction_results[model_name]['unrelaxed_protein']
            else:
                unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
                if not os.path.exists(unrelaxed_pdb_path):
                    logging.error(f"Cannot find {unrelaxed_pdb_path} for relaxation! Skipping...")
                    continue
                unrelaxed_pdb_string = open(unrelaxed_pdb_path, 'r').read()
                unrelaxed_protein = protein.from_pdb_string(unrelaxed_pdb_string)
            relaxed_pdb_str, _, violations = amber_relaxer.process(
                prot=unrelaxed_protein)
            relax_metrics[model_name] = {
                'remaining_violations': violations,
                'remaining_violations_count': sum(violations)
            }
            timings[f'relax_{model_name}'] = time.time() - t_0
            relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
            with open(relax_metrics_path, 'w') as f:
                f.write(json.dumps(relax_metrics, indent=4))
            relaxed_pdbs[model_name] = relaxed_pdb_str
            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(
                output_dir, f'relaxed_{model_name}.pdb')
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)

        with open(timings_path, 'w') as f:
            f.write(json.dumps(timings, indent=4))
              # Write out PDBs in rank order.
        for idx, model_name in enumerate(ranked_order):
            if model_name in relaxed_pdbs:
                protein_instance = relaxed_pdbs[model_name]
            else:
                protein_instance = protein.to_pdb(
                    prediction_results[model_name]['unrelaxed_protein'])
            ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
            with open(ranked_output_path, 'w') as f:
                f.write(protein_instance)
        # Extract multimeric template if multimeric mode is enabled.
        # if multimeric_mode:
        #     feature_dict = multimeric_object.feature_dict
        #     template_mask = feature_dict["template_all_atom_mask"][0]
        #     template_protein = protein.Protein(
        #         atom_positions=feature_dict["template_all_atom_positions"][0],
        #         atom_mask=template_mask,
        #         aatype=feature_dict["template_aatype"][0],
        #         residue_index=feature_dict.get("residue_index", None),
        #         chain_index=feature_dict["asym_id"],
        #         b_factors=np.zeros(template_mask.shape, dtype=float),
        #     )
        #     logging.info(f"template_aatype: {feature_dict['template_aatype'][0]}")
        #     pdb_string = protein.to_pdb(template_protein)
        #     # Check RMSD between the predicted model and the multimeric template.
        #     with tempfile.TemporaryDirectory() as temp_dir:
        #         template_file_path = f"{temp_dir}/template.pdb"
        #         with open(template_file_path, "w") as file:
        #             file.write(pdb_string)
        #         # TODO: use template_sequence for alignment
        #         calculate_rmsd_and_superpose(
        #             template_file_path, ranked_output_path, temp_dir
        #         )

        # Call convert_to_modelcif script
        if convert_to_modelcif:
            parent_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
            logging.info(f"Converting {output_dir} to ModelCIF format...")
            command = [sys.executable,
                       os.path.join(parent_dir, "scripts", "convert_to_modelcif.py"),
                       "--ap_output", output_dir]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"ModelCIF conversion failed ({result.returncode}): "
                              f"{result.stderr or result.stdout}")
            else:
                logging.info("All PDBs converted to ModelCIF format.")
        post_prediction_process(
           output_dir,
           compress_pickles=compress_pickles,
           remove_pickles=remove_pickles,
           remove_keys=remove_keys_from_pickles,
           storage_mode=storage_mode,
        )
