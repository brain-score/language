.. _aws_s3_access_tutorial:

**********************
AWS S3 Access Tutorial
**********************

In order to make data assemblies accessible for Brain-Score model evaluations, they must be be uploaded. 
You can self-host your data (e.g. on S3/OSF), or contact us to host your data on S3. You can also choose to keep your data private 
such that models can be scored, but the data cannot be accessed.
This tutorial walks through the process of uploading and downloading files to S3, and builds off of the information found in
`the docs of BrainIO <https://github.com/brain-score/brainio/blob/main/docs/aws_access.md>`_.


1. Configuration
==========================

Although AWS credentials are unnecessary when downloading public files from S3 (just ask an admin for the information associated with the desired
data assembly), AWS credentials are required when accessing private files or uploading files to S3. Ask an admin to create a user for you 
and they will send you the credentials  necessary to configure your AWS access.

After you have received your credentials, you can configure your access via command line, or via file (see `aws_access <https://github.com/brain-score/brainio/blob/main/docs/aws_access.md>`_). 


2. Uploading Data
===========================

You can upload a new data assembly by using the :code:`upload_data_assembly` method present in the
`S3 util file <https://github.com/brain-score/language/blob/main/brainscore_language/utils/s3.py>`_. The only information
needed to upload the data assembly are the assembly itself, and an assembly identifier that will be used
to name the assembly within S3.

Once uploaded, make note of the hash and version id associated with the upload in order to access the information at a 
later point. (Contact an admin if this information is lost/forgotten, and they can retrieve the information for you!) After 
uploading your data assembly, contact an admin to make your data assembly **public**, which will allow you and others to access
the data assembly easily. 


3. Downloading Data
======================

When downloading a data assembly from S3, it must be made **public** by an admin. You can download data 
assemblies stored in S3 by using the :code:`load_from_s3` method found in the
`S3 util file <https://github.com/brain-score/language/blob/main/brainscore_language/utils/s3.py>`_. The only information
needed to download the data assembly are the :code:`identifier`, :code:`version_id`, and :code:`sha1`,
which are all outputs if you uploaded the data assembly yourself, or can be received by contacting an admin.

