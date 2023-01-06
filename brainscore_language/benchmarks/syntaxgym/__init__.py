from pathlib import Path
import json
from brainscore_language import benchmark_registry
from .benchmark import SyntaxGymSingleTSE, SyntaxGym2020

with open(Path(__file__).parent / 'test_suites.json') as json_file:
    test_suite_dict = json.load(json_file)

benchmark_registry['syntaxgym-2020'] = lambda: SyntaxGym2020()
benchmark_registry['center_embed'] = lambda: SyntaxGymSingleTSE(test_suite_dict['center_embed'])
benchmark_registry['center_embed_mod'] = lambda: SyntaxGymSingleTSE(test_suite_dict['center_embed_mod'])
benchmark_registry['cleft'] = lambda: SyntaxGymSingleTSE(test_suite_dict['cleft'])
benchmark_registry['cleft_modifier'] = lambda: SyntaxGymSingleTSE(test_suite_dict['cleft_modifier'])
benchmark_registry['fgd_hierarchy'] = lambda: SyntaxGymSingleTSE(test_suite_dict['fgd_hierarchy'])
benchmark_registry['fgd_object'] = lambda: SyntaxGymSingleTSE(test_suite_dict['fgd_object'])
benchmark_registry['fgd_pp'] = lambda: SyntaxGymSingleTSE(test_suite_dict['fgd_pp'])
benchmark_registry['fgd_subject'] = lambda: SyntaxGymSingleTSE(test_suite_dict['fgd_subject'])
benchmark_registry['mvrr'] = lambda: SyntaxGymSingleTSE(test_suite_dict['mvrr'])
benchmark_registry['mvrr_mod'] = lambda: SyntaxGymSingleTSE(test_suite_dict['mvrr_mod'])
benchmark_registry['npi_orc_any'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npi_orc_any'])
benchmark_registry['npi_orc_ever'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npi_orc_ever'])
benchmark_registry['npi_src_any'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npi_src_any'])
benchmark_registry['npi_src_ever'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npi_src_ever'])
benchmark_registry['npz_ambig'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npz_ambig'])
benchmark_registry['npz_ambig_mod'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npz_ambig_mod'])
benchmark_registry['npz_obj'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npz_obj'])
benchmark_registry['npz_obj_mod'] = lambda: SyntaxGymSingleTSE(test_suite_dict['npz_obj_mod'])
benchmark_registry['number_orc'] = lambda: SyntaxGymSingleTSE(test_suite_dict['number_orc'])
benchmark_registry['number_prep'] = lambda: SyntaxGymSingleTSE(test_suite_dict['number_prep'])
benchmark_registry['number_src'] = lambda: SyntaxGymSingleTSE(test_suite_dict['number_src'])
benchmark_registry['reflexive_orc_fem'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_orc_fem'])
benchmark_registry['reflexive_orc_masc'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_orc_masc'])
benchmark_registry['reflexive_prep_fem'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_prep_fem'])
benchmark_registry['reflexive_prep_masc'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_prep_masc'])
benchmark_registry['reflexive_src_fem'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_src_fem'])
benchmark_registry['reflexive_src_masc'] = lambda: SyntaxGymSingleTSE(test_suite_dict['reflexive_src_masc'])
benchmark_registry['subordination'] = lambda: SyntaxGymSingleTSE(test_suite_dict['subordination'])
benchmark_registry['subordination_orc-orc'] = lambda: SyntaxGymSingleTSE(test_suite_dict['subordination_orc-orc'])
benchmark_registry['subordination_pp-pp'] = lambda: SyntaxGymSingleTSE(test_suite_dict['subordination_pp-pp'])
benchmark_registry['subordination_src-src'] = lambda: SyntaxGymSingleTSE(test_suite_dict['subordination_src-src'])




