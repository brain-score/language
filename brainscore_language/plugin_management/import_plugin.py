import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class ImportPlugin:
    """ import plugin and (optionally) install dependencies """
    def __init__(self, plugin_type: str, identifier: str):

        self.plugin_type = plugin_type
        self.plugins_dir = Path(__file__).parent.with_name(plugin_type)
        self.identifier = identifier
        self.plugin_dirname = self.locate_plugin()
        self.install_requirements()
        __import__(f'brainscore_language.{self.plugin_type}.{self.plugin_dirname}')


    def locate_plugin(self) -> str:
        """ 
        Searches all plugin_type __init.py__ files for identifier.
        If a match is found of format {plugin_type}_registry[{identifier}],
        returns name of directory where __init.py__ is located 
        """
        plugins = [d.name for d in self.plugins_dir.iterdir() if d.is_dir()]

        specified_plugin_dirname = None
        plugin_registrations_count = 0
        for plugin_dirname in plugins:
            plugin_dirpath = self.plugins_dir / plugin_dirname
            init_file = plugin_dirpath / "__init__.py"
            with open(init_file) as f:
                registry_name = self.plugin_type.strip('s') + '_registry'
                plugin_registrations = [line for line in f if f"{registry_name}['{self.identifier}']"
                                        in line.replace('\"', '\'')]
                if len(plugin_registrations) > 0:
                    specified_plugin_dirname = plugin_dirname
                    plugin_registrations_count += 1

        assert plugin_registrations_count > 0, f"No registrations found for {self.identifier}"
        assert plugin_registrations_count == 1, f"More than one registration found for {self.identifier}"

        return specified_plugin_dirname


    def install_requirements(self):
        requirements_file = self.plugins_dir / self.plugin_dirname / 'requirements.txt'
        if requirements_file.is_file():
            subprocess.run(f"pip install -r {requirements_file}", shell=True)
        else:
            logger.debug(f"Plugin {self.plugin_dirname} has no requirements file {requirements_file}")
