__version__ = "2.9.1.dev0"

def _add_missing_ccd_one_letter_map():
    """Give an older alphafold.common.residue_constants the CCD map templates need.

    alphafold.common comes from ColabFold's alphafold (needed for the fused
    kernels), while alphafold.data comes from the AlphaFold submodule. The
    submodule's mmcif_parsing reads residue_constants.CCD_NAME_TO_ONE_LETTER,
    which ColabFold's residue_constants lacks, so every template mmCIF failed
    to parse. Fall back to the Biopython map AlphaFold used before.
    """
    try:
        from alphafold.common import residue_constants
    except ImportError:
        return
    if hasattr(residue_constants, "CCD_NAME_TO_ONE_LETTER"):
        return
    try:
        from Bio.Data.SCOPData import protein_letters_3to1 as mapping
    except ImportError:
        from Bio.Data.PDBData import protein_letters_3to1_extended
        mapping = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
    residue_constants.CCD_NAME_TO_ONE_LETTER = dict(mapping)


_add_missing_ccd_one_letter_map()
