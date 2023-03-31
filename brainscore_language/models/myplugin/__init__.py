import json
import os
import shutil
import sys
import tempfile
import zipfile
from zipfile import ZipFile


def check_files(config_file, zip=''):
    with open(config_file) as file:
        configs = json.load(file)
    print(configs)
    assert (configs['user_id'] is not None)
    if 'model_type' in configs:
        assert (configs['model_type'] is not None)
        assert (configs['public'] is not None)
        check_zip_file(zip)
    else:
        assert (configs['model_ids'] is not None)


def check_zip_file(zip_file):
    work_dir = tempfile.mkdtemp()
    zip_file = zip_file if os.path.isfile(zip_file) else os.path.realpath(zip_file)

    # remove __MACOSX from zip file:
    zin = zipfile.ZipFile (zip_file, 'r')
    zout = zipfile.ZipFile ('new.zip', 'w')
    for item in zin.infolist():
        buffer = zin.read(item.filename)
        if not str(item.filename).startswith('__MACOSX/'):
            zout.writestr(item, buffer)
    zin.close()

    try:
        zout.extractall(work_dir)
        print(os.listdir(work_dir))
        print(work_dir)
        assert (len(os.listdir(work_dir)) == 1)
    finally:
        shutil.rmtree(work_dir)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        check_files(sys.argv[1], sys.argv[2])
    else:
        check_files(sys.argv[1])

