""" Make plugin details available to readthedocs """

import bibtexparser
import json
import logging
from pathlib import Path
import re
from rstcloth import RstCloth
from typing import Callable, Dict, List, Union

from brainscore_language import benchmark_registry, data_registry, metric_registry, model_registry

_logger = logging.getLogger(__name__)

BIBS_DIR = 'docs/source/bibtex/'
PLUGINS_DOC_FP = 'docs/source/modules/plugins.rst'
PLUGINS_LIST_FP = 'brainscore_language/plugin_management/all_plugins.json'
PLUGIN_DIRS = ['benchmarks', 'data', 'metrics', 'models']
PLUGIN_TYPE_MAP = {plugin_dirtype.strip('s'):plugin_dirtype for plugin_dirtype in PLUGIN_DIRS}


def import_all_plugins():
    for plugin_dirtype in PLUGIN_DIRS:
        plugins_dir = Path(Path(__file__).parents[1], plugin_dirtype)
        for plugin in plugins_dir.glob('[!._]*'):
            if plugin.is_dir():
                plugin_module_path = str(plugin).replace('/', '.')
                __import__(plugin_module_path)

def register_all_plugins() -> Dict[str, Dict]:
    """Add all plugins to respective type registries

    Returns a dict where key is plugin type,
    value is corresponding plugin registry
    """
    postfix = '_registry'
    registries = {PLUGIN_TYPE_MAP[k.strip(postfix)]:globals()[k] 
                    for k in globals().keys() if postfix in k}

    return registries

def check_for_new_plugins(plugins:Dict[str, Callable], plugin_type:str) -> List[str]:
    """ returns a list of plugin names not present in record (all_plugins.json) """
    try:
        with open(PLUGINS_LIST_FP, 'r') as f:
            all_recorded_plugins = json.load(f)
            type_recorded_plugins = all_recorded_plugins[plugin_type]
            new_plugins_list = list(set(plugins) - set(type_recorded_plugins))
    except:
        new_plugins_list = list(plugins)

    return new_plugins_list

def get_plugin_info(plugins:Dict[str, Callable]) -> Dict[str, Dict]:
    """Retrieves information about the plugin from the plugin module

    Returns a dict where key is plugin name,
    value is dict that includes:

    bibtex: a bibtex string
    bibtex_id: bibtex identifier
    """
    plugin_info = {}
    for plugin_name in plugins:
        plugin = plugins[plugin_name]()
        if hasattr(plugin, 'bibtex'):
            plugin_info[plugin_name] = {
                'bibtex':plugin.bibtex,
                'bibtex_id':_id_from_bibtex(plugin.bibtex)
            }
        else:
            plugin_info[plugin_name] = {}

    return plugin_info

def _id_from_bibtex(bibtex:str) -> str:
    return re.search('\{(.*?),', bibtex).group(1)

def _remove_duplicate_bibs(plugin_info:Dict[str,Dict], plugins_bib_fp:str) -> List[str]:
    """ ensure no duplicate BibTeX entries """
    # remove duplicate BibTeX within new plugins
    new_plugin_bibtex = set([v['bibtex'] for k,v in plugin_info.items()])
    # don't add new BibTeX if same ID as already recorded BibTeX
    try:
        with open(plugins_bib_fp, 'r') as bibfile:
            bibtex_db = bibtexparser.load(bibfile)
            entries = bibtex_db.get_entry_list()
            existing_ids = set([entry['ID'] for entry in entries])
        new_ids = set([_id_from_bibtex(b) for b in new_plugin_bibtex])
        duplicates = existing_ids & new_ids
        ids_to_add = new_ids - duplicates
        bibtex_to_add = {b for b in new_plugin_bibtex if _id_from_bibtex(b) in ids_to_add}
    except:
        bibtex_to_add = [b for b in new_plugin_bibtex]

    return bibtex_to_add

def _record_bibtex(bibtex_to_add:List[str], plugins_bib_fp:str):
    """ insert new BibTeX into respective .bib files """
    if plugins_bib_fp.exists():
        open_condition = "a"
    else:
        open_condition = "w+"
    with open(plugins_bib_fp, open_condition) as f:
        if plugins_bib_fp.stat().st_size > 0:
            f.write('\n')
        for i, bibtex in enumerate(bibtex_to_add):
            f.write(bibtex)
            f.write('\n')

def add_new_bibtex(new_plugins:Dict[str,str], plugin_type='refs'):
    """ For all new plugins, add bibtex (if present) to .bib files """
    if plugin_type == 'refs':
        new_plugins = dict(ele for sub in new_plugins.values() for ele in sub.items())
    # drop plugins without bibtex
    plugins_with_bibtex = {k:v for k,v in new_plugins.items() if 'bibtex' in v.keys()}
    if len(plugins_with_bibtex.keys()) > 0:
        plugins_bib_fp = Path(BIBS_DIR + plugin_type + '.bib')
        # add bibtex (if present) to .bib files
        bibtex_to_add = _remove_duplicate_bibs(plugins_with_bibtex, plugins_bib_fp)
        _record_bibtex(bibtex_to_add, plugins_bib_fp)

def _stripped_line(line:str) -> str:
    """Returns plugin type match to rst type header (e.g. '_Benchmarks\n' -> benchmarks) """
    return line.strip().lower()

def _parse_rst() -> Dict[str, List[List[str]]]:
    """Reads plugin info from plugins.rst into dict

    Returns a dict where key is plugin type,
    value is a list of lists containing plugin info strings
    """
    with open(PLUGINS_DOC_FP, 'r') as f:
        lines = f.readlines()
        headings = {l:i for i, l in enumerate(lines) if _stripped_line(l) in PLUGIN_DIRS or _stripped_line(l) == 'bibliography'}
        plugins = {}
        for i, line in enumerate(lines):
            if i in headings.values() and i != max(headings.values()):
                j = i+2
                next_header_idx = min([head_idx for head_idx in headings.values() if head_idx > i])
                current_plugin = []
                while j < next_header_idx:
                    if not lines[j].startswith('\n'):
                        current_plugin.append(lines[j])
                    else:
                        if line in plugins:
                            plugins[line].append(current_plugin)
                        else:
                            plugins[line] = [current_plugin]
                        current_plugin = []
                    j+=1

    return plugins

def _infotype_idx(info_list:List[str], search_string:str) -> int:
    """ Returns index of plugin info string that matches search_string """
    return [i for i in info_list if search_string in i][0]

def _clean_existing_plugins(existing_plugins:Dict[str, List[List]]) -> Dict[str,Dict[str,Dict[str,str]]]:
    """Cleans and formats dict of plugins read from plugins.rst

    Returns a dict where key is plugin type, value is a dict
    of plugin names (str) mapped to a dict of their info strings

    NOTE: info is currently just BiBTeX citation, but could expand
    e.g. include description
    """
    cleaned_existing_plugins = {}
    for plugin_type in existing_plugins:
        cleaned_plugin_info = [[info.strip() for info in l] for l in existing_plugins[plugin_type]]
        cleaned_existing_plugins[plugin_type.strip()] = {info[0]:({'citation':info[_infotype_idx(info,':cite:')]} 
                                                            if len(info)>2 else {'citation':None}) 
                                                            for info in cleaned_plugin_info}

    return cleaned_existing_plugins

def _clean_new_plugins(new_plugins:Dict[str, List[List]]) -> Dict[str, List[List]]:
    """Cleans and formats dict of new plugins

    Returns a dict where key is plugin type, value is a dict
    of plugin names (str) mapped to a dict of their info strings

    NOTE: info is currently just BiBTeX citation, but could expand
    e.g. include description
    """
    cleaned_new_plugins = {}
    for plugin_type in new_plugins:
        cleaned_new_plugins[plugin_type.capitalize()] = {k:({'citation':':cite:label:`' + v['bibtex_id'] +'`'}
                                                            if 'bibtex_id' in v.keys() else {'citation':None}) 
                                                            for k,v in new_plugins[plugin_type].items()}
    
    return cleaned_new_plugins

def _write_to_rst(all_plugins:Dict[str,Dict]):
    """ Writes plugin info to readthedocs plugins.rst """
    with open(PLUGINS_DOC_FP, 'w+') as f:
        doc = RstCloth(f)
        doc.ref_target(name="plugins")
        doc.title('Plugins')
        doc.newline()
        for plugin_type in all_plugins:
            doc.h3(plugin_type)
            for plugin in all_plugins[plugin_type]:
                doc.h4(plugin)
                if all_plugins[plugin_type][plugin]['citation']:
                    doc.content(all_plugins[plugin_type][plugin]['citation'])
                doc.newline()
        doc.h2('Bibliography')
        doc.directive(name="bibliography", fields=[('all','')])


def update_readthedocs(new_plugin_info:Dict[str,Dict]):
    """ For all new plugins, add name and info to readthedocs (plugins.rst) """
    # clean and format data
    cleaned_new_plugins = _clean_new_plugins(new_plugin_info)
    try:
        existing_plugins = _parse_rst()
        cleaned_existing_plugins = _clean_existing_plugins(existing_plugins)
        # add new plugin info to existing plugin info
        all_plugins = cleaned_existing_plugins
        for k,v in cleaned_new_plugins.items():
            if k in all_plugins:
                all_plugins[k].update(v)
            else:
                all_plugins[k] = v
    except:
        all_plugins = cleaned_new_plugins

    # write plugin info to readthedocs
    _write_to_rst(all_plugins)

def update_plugins_list(all_plugins):
    """ Update master list to include any new plugins """
    plugins_lists = {k:list(v) for k,v in all_plugins.items()}
    with open(PLUGINS_LIST_FP, 'w+') as f:
        json.dump(plugins_lists, f, indent=2) 

if __name__ == '__main__':
    import_all_plugins()
    all_plugins = register_all_plugins()
    new_plugin_info = {}
    for plugin_type in all_plugins:
        new_plugins_list = check_for_new_plugins(all_plugins[plugin_type], plugin_type)
        if len(new_plugins_list) > 0:
            new_plugins = {k:v for k,v in all_plugins[plugin_type].items() if k in new_plugins_list}
            print(f'The following new {plugin_type} have been identified:')
            print(*new_plugins, sep='\n')
            plugin_info = get_plugin_info(new_plugins)
            new_plugin_info[plugin_type] = plugin_info
            add_new_bibtex(plugin_info, plugin_type) # plugin type .bib file
        else:
            print('No new plugins identified.')
    add_new_bibtex(new_plugin_info) # one .bib file to rule them all
    update_readthedocs(new_plugin_info)
    update_plugins_list(all_plugins)
