from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from six.moves.urllib.parse import urlparse
from tqdm import tqdm

import brainio.assemblies as assemblies
import brainio.stimuli as stimuli
from brainio.stimuli import StimulusSetLoader
from brainio.lookup import lookup_assembly, lookup_stimulus_set, sha1_hash

BRAINIO_HOME = 'BRAINIO_HOME'

_logger = logging.getLogger(__name__)
_local_data_path = None


def get_local_data_path():
    # This makes it easier to mock.
    global _local_data_path
    if _local_data_path is None:
        _local_data_path = os.path.expanduser(os.getenv(BRAINIO_HOME, '~/.brainio'))
    return _local_data_path


class Fetcher(object):
    """A Fetcher obtains data with which to populate a DataAssembly.  """

    def __init__(self, location, local_filename):
        self.location = location
        self.local_filename = local_filename
        self.local_dir_path = os.path.join(get_local_data_path(), self.local_filename)
        os.makedirs(self.local_dir_path, exist_ok=True)

    def fetch(self):
        """
        Fetches the resource identified by location.
        :return: a full local file path
        """
        raise NotImplementedError("The base Fetcher class does not implement .fetch().  Use a subclass of Fetcher.")


class BotoFetcher(Fetcher):
    """A Fetcher that retrieves files from Amazon Web Services' S3 data storage.  """

    def __init__(self, location, local_filename):
        super(BotoFetcher, self).__init__(location, local_filename)
        parsed_url = urlparse(self.location)
        split_path = parsed_url.path.lstrip('/').split("/")
        # http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#access-bucket-intro
        virtual_hosted_style = 's3.' in parsed_url.hostname  # s3. for virtual hosted style; s3- for older AWS
        if virtual_hosted_style:
            self.bucketname = parsed_url.hostname.split(".s3.")[0]
            self.relative_path = os.path.join(*(split_path))
        else:
            self.bucketname = split_path[0]
            self.relative_path = os.path.join(*(split_path[1:]))
        self.output_filename = os.path.join(self.local_dir_path, self.relative_path)
        self._logger = logging.getLogger(fullname(self))

    def fetch(self):
        if not os.path.exists(self.output_filename):
            self.download_boto()
        return self.output_filename

    def download_boto(self):
        """Downloads file from S3 via boto at `url` and writes it in `self.output_filename`."""
        self._logger.info('downloading %s' % self.relative_path)
        try:  # try with authentication
            self._logger.debug("attempting default download (signed)")
            self.download_boto_config(config=None)
        except Exception as e_signed:  # try without authentication
            self._logger.debug("default download failed, trying unsigned")
            # disable signing requests. see https://stackoverflow.com/a/34866092/2225200
            unsigned_config = Config(signature_version=UNSIGNED)
            try:
                self.download_boto_config(config=unsigned_config)
            except Exception as e_unsigned:
                # when unsigned download also fails, raise both exceptions
                # raise Exception instead of specific type to avoid missing __init__ arguments
                raise Exception([e_signed, e_unsigned])

    def download_boto_config(self, config):
        s3 = boto3.resource('s3', config=config)
        obj = s3.Object(self.bucketname, self.relative_path)
        # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
        with tqdm(total=obj.content_length, unit='B', unit_scale=True,
                  desc=self.bucketname + "/" + self.relative_path) as progress_bar:
            def progress_hook(bytes_amount):
                if bytes_amount > 0:  # at the end, this sometimes passes a negative byte amount which tqdm can't handle
                    progress_bar.update(bytes_amount)

            obj.download_file(self.output_filename, Callback=progress_hook)


def verify_sha1(filepath, sha1):
    actual_hash = sha1_hash(filepath)
    if sha1 != actual_hash:
        raise IOError(f"File '{filepath}': invalid SHA-1 hash {actual_hash} (expected {sha1})")
    _logger.debug(f"sha1 OK: {filepath}")


_fetcher_types = {
    "S3": BotoFetcher,
}


def get_fetcher(type="S3", location=None, local_filename=None):
    return _fetcher_types[type](location, local_filename)


def fetch_file(location_type, location, sha1):
    filename = filename_from_link(location)
    fetcher = get_fetcher(type=location_type, location=location,
                          local_filename=filename)
    local_path = fetcher.fetch()
    verify_sha1(local_path, sha1)
    return local_path


def filename_from_link(location):
    parse = urlparse(location)
    local_name = os.path.basename(parse.path)
    local_name = os.path.splitext(local_name)[0]
    return local_name


def unzip(zip_path):
    containing_dir = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        if not all(map(lambda filename: os.path.exists(os.path.join(containing_dir, filename)), zip_file.namelist())):
            _logger.debug(f"Extractall to {containing_dir}")
            zip_file.extractall(containing_dir)
    return containing_dir


def resolve_assembly_class(class_name):
    cls = getattr(assemblies, class_name)
    return cls


def resolve_stimulus_set_class(class_name):
    cls = getattr(stimuli, class_name)
    return cls


def get_assembly(identifier):
    assembly_lookup = lookup_assembly(identifier)
    file_path = fetch_file(location_type=assembly_lookup['location_type'],
                            location=assembly_lookup['location'], sha1=assembly_lookup['sha1'])
    stimulus_set = get_stimulus_set(assembly_lookup['stimulus_set_identifier'])
    cls = resolve_assembly_class(assembly_lookup['class'])
    loader = cls.get_loader_class()(
        cls=cls,
        file_path=file_path,
        stimulus_set_identifier=assembly_lookup['stimulus_set_identifier'],
        stimulus_set=stimulus_set,
    )
    assembly = loader.load()
    assembly.attrs['identifier'] = identifier
    return assembly


def get_stimulus_set(identifier):
    csv_lookup, zip_lookup = lookup_stimulus_set(identifier)
    csv_path = fetch_file(location_type=csv_lookup['location_type'], location=csv_lookup['location'],
                          sha1=csv_lookup['sha1'])
    zip_path = fetch_file(location_type=zip_lookup['location_type'], location=zip_lookup['location'],
                          sha1=zip_lookup['sha1'])
    stimuli_directory = unzip(zip_path)
    loader = StimulusSetLoader(
        csv_path=csv_path,
        stimuli_directory=stimuli_directory,
        cls=resolve_stimulus_set_class(csv_lookup['class'])
    )
    stimulus_set = loader.load()
    stimulus_set.identifier = identifier
    # ensure perfect overlap
    stimuli_paths = [Path(stimuli_directory) / local_path for local_path in os.listdir(stimuli_directory)
                     if not local_path.endswith('.zip') and not local_path.endswith('.csv')]
    assert set(stimulus_set.stimulus_paths.values()) == set(stimuli_paths), \
        "Inconsistency: unzipped stimuli paths do not match csv paths"
    return stimulus_set


def fullname(obj):
    return obj.__module__ + "." + obj.__class__.__name__
