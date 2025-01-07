#!/usr/bin/env python
# coding=utf-8
"""
upload script
"""
import os
import sys
import multiprocessing

# 从Python SDK导入BOS配置管理模块以及安全认证模块
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce import exception
from baidubce.services import bos
from baidubce.services.bos import canned_acl
from baidubce.services.bos.bos_client import BosClient


# 设置BosClient的Host，Access Key ID和Secret Access Key
bos_host = "bj.bcebos.com"
access_key_id = os.getenv("AK")
secret_access_key = os.getenv("SK")
# 创建BceClientConfiguration
config = BceClientConfiguration(credentials=BceCredentials(access_key_id, secret_access_key), endpoint=bos_host)

bucket_name = sys.argv[2]
object_key = sys.argv[1]
file_name = sys.argv[1]

bos_client = BosClient(config)
upload_id = bos_client.initiate_multipart_upload(bucket_name, object_key).upload_id

fsize = os.path.getsize(object_key)
fsize = fsize / float(1024 * 1024)
if fsize < 200:
    chunk_size = 10
else:
    chunk_size = 100

result = bos_client.put_super_obejct_from_file(
    bucket_name, object_key, file_name, chunk_size=100, thread_num=multiprocessing.cpu_count()
)
if result:
    print("Upload success!")
