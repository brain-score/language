""" Make plugin details available to readthedocs """

import json
import logging
from pathlib import Path
import re
from rstcloth import RstCloth
from typing import Dict, List, Union

BIBS_DIR = 'docs/source/bibtex/'
PLUGINS_DOC_FP = 'docs/source/modules/plugins.rst'
PLUGINS_LIST_FP = 'brainscore_language/plugin_management/all_plugins.json'
PLUGIN_DIRS = ['benchmarks', 'data', 'metrics', 'models']
PLUGIN_TYPE_MAP = {plugin_dirtype:plugin_dirtype.strip('s') for plugin_dirtype in PLUGIN_DIRS}


def _get_module_plugin_names(plugin_type:str, plugin_dir:Path) -> List[str]:
    """ Returns list of plugins registered by module """
    init_fp = plugin_dir / "__init__.py"
    registry = PLUGIN_TYPE_MAP[plugin_type] + "_registry"

    with open(init_fp, 'r') as f:
        text = f.read()
        registered_plugins = re.findall(registry+'\[(.*)\]', text)
        cleaned_plugin_names = [name.replace('"', '').replace('\'', '') for name in registered_plugins]
    
    return cleaned_plugin_names

def _id_from_bibtex(bibtex:str) -> str:
    """ Returns BibTeX identifier from BibTeX """
    return re.search('\{(.*?),', bibtex).group(1)

def get_all_plugin_info() -> Dict[str, Dict[str, Dict[str, Union[list, str, str]]]]:
    """Add all plugins to respective type registries

    Returns a dict where key is plugin type,
    value is a dict where key is name of plugin dir,
    value is a dict of plugin info:

    plugin_names: list of names of all plugins registered by module
    bibtex: a BibTeX string
    bibtex_id: BibTeX identifier
    """
    all_plugin_info = {}
    for plugin_type in PLUGIN_DIRS:
        plugins_dir = Path(Path(__file__).parents[1], plugin_type)
        for plugin_dir in plugins_dir.glob('[!._]*'):
            plugin_dirname = str(plugin_dir)
  
            if plugin_type not in all_plugin_info:
                all_plugin_info[plugin_type] = {plugin_dirname:{}}
            else:
                all_plugin_info[plugin_type].update({plugin_dirname:{}})

            plugin_dir_dict = all_plugin_info[plugin_type][plugin_dirname]

            plugin_names = _get_module_plugin_names(plugin_type, plugin_dir)
            plugin_module_path = plugin_dirname.replace('/', '.')
            plugin_module = __import__(plugin_module_path, fromlist=['BIBTEX'])

            plugin_dir_dict['plugin_names'] = plugin_names
            if hasattr(plugin_module, 'BIBTEX'):
                plugin_dir_dict['bibtex'] = plugin_module.BIBTEX
                plugin_dir_dict['bibtex_id'] = _id_from_bibtex(plugin_module.BIBTEX) 

    return all_plugin_info

def _remove_duplicate_bibs(plugins_with_bibtex=Dict[str, Dict]):
    """ Returns list of unique BibTeX to add """
    bibtex_data = {v['bibtex_id']:v['bibtex'] for v in plugins_with_bibtex.values()}
    alphabetized_bibtex = dict(sorted(bibtex_data.items()))
    deduped_bibtex = list(alphabetized_bibtex.values())

    return deduped_bibtex

def _record_bibtex(bibtex_to_add:List[str], plugins_bib_fp:str):
    """ insert new BibTeX into respective .bib files """
    if not Path(BIBS_DIR).exists():
        Path(BIBS_DIR).mkdir(parents=True)
    with open(plugins_bib_fp, "w+") as f:
        for bibtex in bibtex_to_add:
            f.write(bibtex)
            f.write('\n')

def create_bibfile(plugins=Dict[str, Dict], plugin_type='refs'):
    """ For all plugins, add bibtex (if present) to .bib files """
    if plugin_type == 'refs':
        plugins = dict(ele for sub in plugins.values() for ele in sub.items())
    # drop plugins without bibtex
    plugins_with_bibtex = {k:v for k,v in plugins.items() if 'bibtex' in v.keys()}
    if len(plugins_with_bibtex.keys()) > 0:
        plugins_bib_fp = Path(BIBS_DIR + plugin_type + '.bib')
        # add bibtex (if present) to .bib files
        bibtex_to_add = _remove_duplicate_bibs(plugins_with_bibtex)
        _record_bibtex(bibtex_to_add, plugins_bib_fp)

def _prepare_content(all_plugin_info:Dict[str, Dict]) -> Dict[str, Dict]:
    """Converts plugin information into rst format

    Returns a dict where key is plugin type, value is a dict
    of plugin names (str) mapped to a dict of their info

    NOTE: info is currently plugin directory paths and BiBTeX citations, 
    but could expand to e.g. include description of plugin
    """
    prepared_plugin_info = {}
    for plugin_type in all_plugin_info:
        plugin_type_title = plugin_type.capitalize()
        prepared_plugin_info[plugin_type_title] = {name:{'dirname':k, 
                                                        'citation':(':cite:label:`' + v['bibtex_id'] +'`'
                                                        if 'bibtex_id' in v.keys() else None)} 
                                                        for k,v in all_plugin_info[plugin_type].items() 
                                                        for name in v['plugin_names']}
    return prepared_plugin_info

def _write_to_rst(plugin_info:Dict[str,Dict]):
    """ Writes plugin info to readthedocs plugins.rst """
    with open(PLUGINS_DOC_FP, 'w+') as f:
        doc = RstCloth(f)
        doc.newline()
        doc.ref_target(name="plugins")
        doc.title('Plugins')
        doc.newline()
        for plugin_type in plugin_info:
            doc.h3(plugin_type)
            for plugin in plugin_info[plugin_type]:
                doc.h4(plugin)
                doc.content(plugin_info[plugin_type][plugin]['dirname'])
                doc.newline()
                if plugin_info[plugin_type][plugin]['citation']:
                    doc.content(plugin_info[plugin_type][plugin]['citation'])
                doc.newline()
        doc.h2('Bibliography')
        doc.directive(name="bibliography", fields=[('all','')])

def update_readthedocs(all_plugin_info:Dict[str,Dict]):
    """ For all plugins, add name and info to readthedocs (plugins.rst) """
    prepared_plugin_info = _prepare_content(all_plugin_info) # rst formatting
    _write_to_rst(prepared_plugin_info)

if __name__ == '__main__':
    all_plugin_info = get_all_plugin_info()
    for plugin_type in all_plugin_info:
        create_bibfile(all_plugin_info[plugin_type], plugin_type) # plugin type .bib file
    create_bibfile(all_plugin_info) # one .bib file to rule them all
    update_readthedocs(all_plugin_info)
