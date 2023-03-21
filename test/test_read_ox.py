"""
A unittest script that test if creating MonomericObject 
or MultimericObject works
"""
import unittest
import os
import pickle
import sys
from alphapulldown.objects import MonomericObject
import importlib
from absl import app
from absl import flags
from absl import logging
import shutil
from alphafold.data.pipeline import DataPipeline
from alphafold.data.tools import hmmsearch
from alphafold.data import templates
from alphapulldown.utils import *
import numpy as np
import os
from absl import logging, app
import numpy as np
from alphapulldown.utils import *
import contextlib
from datetime import datetime
import alphafold
from pathlib import Path
from colabfold.utils import DEFAULT_API_SERVER

class TestParser(unittest.TestCase):
    def setUp(self) -> None:
        self.jackhmmer_binary_path = shutil.which("jackhmmer")
        self.hmmsearch_binary_path = shutil.which("hmmsearch")
        self.hhblits_binary_path = shutil.which("hhblits")
        self.kalign_binary_path = shutil.which("kalign")
        self.hmmbuild_binary_path = shutil.which("hmmbuild")
        self.fasta_paths = "./example_data/P03452.fasta"
        self.monomer_object_dir = "./example_data/"
        self.output_dir = "./example_data/"
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.0/"
        self.max_template_date = "2200-01-01"
        self.oligomer_state_file = "./example_data/oligomer_state_file.txt"
        self.custom = "./example_data/custom.txt"
        self.uniref30_database_path = os.path.join(
            self.data_dir, "uniref30", "UniRef30_2021_03"
        )
        self.uniref90_database_path = os.path.join(
            self.data_dir, "uniref90", "uniref90.fasta"
        )
        
        self.mgnify_database_path = os.path.join(
            self.data_dir, "mgnify", "mgy_clusters_2022_05.fa"
        )
        self.bfd_database_path = os.path.join(
            self.data_dir,
            "bfd",
            "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        )
        self.small_bfd_database_path = os.path.join(
            self.data_dir, "small_bfd", "bfd-first_non_consensus_sequences.fasta"
        )

        self.pdb_seqres_database_path = os.path.join(
            self.data_dir, "pdb_seqres", "pdb_seqres.txt"
        )

        self.template_mmcif_dir = os.path.join(self.data_dir, "pdb_mmcif", "mmcif_files")

        self.obsolete_pdbs_path = os.path.join(self.data_dir, "pdb_mmcif", "obsolete.dat")
  
        self.pdb70_database_path = os.path.join(self.data_dir, "pdb70", "pdb70")

        self.extra_msa_db_path = "./example_data/extra_msa_db_cleaned.fasta"

        return super().setUp()
    
    def test_1_initialise_MonomericObject(self):
        """Test initialise a monomeric object"""
        sequences, descriptions = parse_fasta(open(self.fasta_paths,'r').read())
        monomer_object = MonomericObject(description=descriptions[0],sequence=sequences[0])
        assert monomer_object.description == descriptions[0]
        assert monomer_object.sequence == sequences[0]
        return monomer_object
    
    def test_2_initialise_datapipeline(self):
        """Test setting up datapipelines"""
        monomer_data_pipeline = DataPipeline(
        jackhmmer_binary_path=self.jackhmmer_binary_path,
        hhblits_binary_path=self.hhblits_binary_path,
        uniref90_database_path=self.uniref90_database_path,
        mgnify_database_path=self.mgnify_database_path,
        bfd_database_path=self.bfd_database_path,
        uniref30_database_path=self.uniref30_database_path,
        small_bfd_database_path=self.small_bfd_database_path,
        use_small_bfd=False,
        use_precomputed_msas=True,
        template_searcher=hmmsearch.Hmmsearch(
            binary_path=self.hmmsearch_binary_path,
            hmmbuild_binary_path=self.hmmbuild_binary_path,
            database_path=self.pdb_seqres_database_path,
        ),
        template_featurizer=templates.HmmsearchHitFeaturizer(
            mmcif_dir=self.template_mmcif_dir,
            max_template_date=self.max_template_date,
            max_hits=20,
            kalign_binary_path=self.kalign_binary_path,
            obsolete_pdbs_path=self.obsolete_pdbs_path,
            release_dates_path=None,
        ),)
        return monomer_data_pipeline
    
    def test_3_add_more_seqdb(self):
        """Test add extra sequence db """
        monomer_obj = self.test_1_initialise_MonomericObject()
        monomer_pipeline = self.test_2_initialise_datapipeline()
        uniprot_database_path = os.path.join(self.data_dir, "uniprot/uniprot.fasta")
        uniprot_runner = create_uniprot_runner(self.jackhmmer_binary_path,uniprot_database_path)
        extra_msa_runner = create_uniprot_runner(self.jackhmmer_binary_path,self.extra_msa_db_path)
        # monomer_pipeline.extra_msa_db_path = self.extra_msa_db_path
        monomer_obj.uniprot_runner=uniprot_runner
        monomer_obj.extra_msa_runner=extra_msa_runner
        print("Now will test make features")
        monomer_obj.make_features(monomer_pipeline,self.output_dir
                                  ,use_precomputed_msa=True,save_msa=False)
        
        pickle.dump(monomer_obj,open(f"{monomer_obj.description}.pkl","wb"))

if __name__ =="__main__":
    unittest.main()