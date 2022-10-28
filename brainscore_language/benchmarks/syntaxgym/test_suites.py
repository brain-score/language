import json
name1 = 'center_embed'
name2 = 'center_embed_mod'
name3 = 'cleft'
name4 = 'cleft_modifier'
name5 = 'fgd-embed3'
name6 = 'fgd-embed4'
name7 = 'fgd_hierarchy'
name8 = 'fgd_object'
name9 = 'fgd_pp'
name10 = 'fgd_subject'
name11 = 'mvrr'
name12 = 'mvrr_mod'
name13 = 'nn-nv-rpl'
name14 = 'npi_orc_any'
name15 = 'npi_orc_ever'
name16 = 'npi_src_any'
name17 = 'npi_src_ever'
name18 = 'npz_ambig'
name19 = 'npz_ambig_mod'
name20 = 'npz_obj'
name21 = 'npz_obj_mod'
name22 = 'number_orc'
name23 = 'number_prep'
name24 = 'number_src'
name25 = 'reflexive_orc_fem'
name26 = 'reflexive_orc_masc'
name27 = 'reflexive_prep_fem'
name28 = 'reflexive_prep_masc'
name29 = 'reflexive_src_fem'
name30 = 'reflexive_src_masc'
name31 = 'subordination'
name32 = 'subordination_orc-orc'
name33 = 'subordination_pp-pp'
name34 = 'subordination_src-src'

suite1 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/center_embed.json'
suite2 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/center_embed_mod.json'
suite3 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/cleft.json'
suite4 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/cleft_modifier.json'
suite5 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd-embed3.json'
suite6 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd-embed4.json'
suite7 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd_hierarchy.json'
suite8 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd_object.json'
suite9 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd_pp.json'
suite10 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/fgd_subject.json'
suite11 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/mvrr.json'
suite12 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/mvrr_mod.json'
suite13 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/nn-nv-rpl.json'
suite14 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npi_orc_any.json'
suite15 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npi_orc_ever.json'
suite16 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npi_src_any.json'
suite17 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npi_src_ever.json'
suite18 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npz_ambig.json'
suite19 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npz_ambig_mod.json'
suite20 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npz_obj.json'
suite21 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/npz_obj_mod.json'
suite22 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/number_orc.json'
suite23 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/number_prep.json'
suite24 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/number_src.json'
suite25 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_orc_fem.json'
suite26 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_orc_masc.json'
suite27 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_prep_fem.json'
suite28 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_prep_masc.json'
suite29 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_src_fem.json'
suite30 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/reflexive_src_masc.json'
suite31 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/subordination.json'
suite32 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/subordination_orc-orc.json'
suite33 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/subordination_pp-pp.json'
suite34 = 'https://raw.githubusercontent.com/cpllab/syntactic-generalization/nextflow/test_suites/json/subordination_src-src.json'

test_suite_dict = {name1: suite1,
name2: suite2,
name3: suite3,
name4: suite4,
name5: suite5,
name6: suite6,
name7: suite7,
name8: suite8,
name9: suite9,
name10: suite10,
name11: suite11,
name12: suite12,
name13: suite13,
name14: suite14,
name15: suite15,
name16: suite16,
name17: suite17,
name18: suite18,
name19: suite19,
name20: suite20,
name21: suite21,
name22: suite22,
name23: suite23,
name24: suite24,
name25: suite25,
name26: suite26,
name27: suite27,
name28: suite28,
name29: suite29,
name30: suite30,
name31: suite31,
name32: suite32,
name33: suite33,
name34: suite34}
        
with open('test_suites.json', 'w') as fp:
    json.dump(test_suite_dict, fp)