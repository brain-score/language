.. _aws_s3_access_tutorial:

**********************
AWS S3 Access Tutorial
**********************

In order to make data assemblies accessible for Brain-Score model evaluations, they must be be uploaded. 
You can self-host your data (e.g. on S3/OSF), or contact us to host your data on S3. You can also choose to keep your data private 
such that models can be scored, but the data cannot be accessed.
This tutorial walks through the process of uploading and downloading files to S3, and builds off of the information found in
`the docs of BrainIO <https://github.com/brain-score/brainio/blob/main/docs/aws_access.md>`_.


1. Configuration (see `aws_access <https://github.com/brain-score/brainio/blob/main/docs/aws_access.md>`_)
==========================

Although AWS credentials are unnecessary when downloading public files from S3 (just ask an admin for the information associated with the desired
data assembly), AWS credentials are required when accessing private files or uploading files to S3. Ask an admin to create a user for you 
and they will send you the credentials  necessary to configure your AWS access.

After you have received your credentials, you can configure your access in two ways:

## Option 1: configuration via command line
Use `awscli` to configure your AWS access.
```shell
pip install awscli
aws configure
```
(use region `us-east-1`, output-format `json`)

## Option 2: configuration via file
You can also create the configuration files yourself.

Create your credentials file at `~/.aws/credentials`:
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

To configure region and output format, create a config file at `~/.aws/config`:
```
[default]
region = us-east-1
output-format = json
```


2. Uploading Data
===========================

You can upload a new data assembly by using the :code:`upload_data_assembly` method present in the
`S3 util file <https://github.com/brain-score/language/blob/main/brainscore_language/utils/s3.py>`_. The only information
needed to upload the data assembly are the assembly itself, and an assembly identifier that will be used
to name the assembly within S3, as seen in the following excerpt.

.. code-block:: python

    import logging
    from pathlib import Path
    from brainio import fetch
    from brainio.packaging import write_netcdf, upload_to_s3

    _logger = logging.getLogger(__name__)


    def upload_data_assembly(assembly, assembly_identifier, bucket_name="brainscore-language", assembly_prefix="assy_"):
        ...

        # identifiers
        assembly_store_identifier = assembly_prefix + assembly_identifier.replace(".", "_")
        netcdf_file_name = assembly_store_identifier + ".nc"
        target_netcdf_path = Path(fetch.get_local_data_path()) / assembly_store_identifier / netcdf_file_name
        s3_key = netcdf_file_name

        # write to disk and upload
        netcdf_kf_sha1 = write_netcdf(assembly, target_netcdf_path)
        response = upload_to_s3(target_netcdf_path, bucket_name, s3_key)
        _logger.debug(f"Uploaded {assembly_store_identifier} to S3 "
                      f"with key={s3_key}, sha1={netcdf_kf_sha1}, version_id={response['VersionId']}: {response}")
        response['sha1'] = netcdf_kf_sha1
        return response

Once uploaded, make note of the hash and version id associated with the upload in order to access the information at a 
later point. (Contact an admin if this information is lost/forgotten, and they can retrieve the information for you!) After 
uploading your data assembly, contact an admin to make your data assembly **public**, which will allow you and others to access
the data assembly easily. 


3. Downloading Data
======================

When downloading a data assembly from S3, it must be made **public** by an admin. You can download data 
assemblies stored in S3 by using the :code:`load_from_s3` method found in the
`s3 util file <https://github.com/brain-score/language/blob/main/brainscore_language/utils/s3.py>`_. The only information
needed to download the data assembly are the :code:`identifier`, :code:`version_id`, and :code:`sha1` (seen in the following excerpt),
which are all outputs if you uploaded the data assembly yourself, or can be received by contacting an admin.

.. code-block:: python

    from brainio.assemblies import AssemblyLoader, NeuroidAssembly, DataAssembly
    from brainio.fetch import fetch_file
    ...


    def load_from_s3(identifier, version_id, sha1, assembly_prefix="assy_", cls=NeuroidAssembly) -> DataAssembly:
        filename = f"{assembly_prefix}{identifier.replace('.', '_')}.nc"
        file_path = fetch_file(location_type="S3",
                               location=f"https://brainscore-language.s3.amazonaws.com/{filename}",
                               version_id=version_id,
                               sha1=sha1)
        loader = AssemblyLoader(cls=cls, file_path=file_path)
        assembly = loader.load()
        assembly.attrs['identifier'] = identifier
        return assembly
