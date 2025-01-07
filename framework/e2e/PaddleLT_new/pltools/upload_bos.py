#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
  * @file:  upload_bos.py
  * @author:  luozeyu01
  * @date  2023/4/12 4:35 PM
  * @brief
  *
  **************************************************************************/
"""

import os
import logging

from baidubce.services.bos.bos_client import BosClient
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials


class UploadBos(object):
    """bos upload tool"""

    def __init__(self):
        self.ak = os.environ.get("AK")
        self.sk = os.environ.get("SK")
        if self.ak is None or self.sk is None:
            raise Exception("sk or ak is None!! please export or set SK and AK !!!")

        bos_host = "bj.bcebos.com"
        logger = logging.getLogger("baidubce.http.bce_http_client")
        fh = logging.FileHandler("fdb_sample.log")
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        config = BceClientConfiguration(credentials=BceCredentials(self.ak, self.sk), endpoint=bos_host)
        self.bos_client = BosClient(config)

    def upload_to_bos(self, bos_path, file_path):
        """
        bos_path: bos上生成的路径 'paddle-qa/PaddleLT/PaddleLTBenchmark/jobs_result'
        file_path: 要上传的文件夹绝对路径
        """
        object_key = "{}".format(os.path.basename(file_path))
        print(object_key, file_path)
        self.bos_client.put_object_from_file(bos_path, object_key, file_path)


if __name__ == "__main__":
    bos_path = "paddle-qa/PaddleLT/PaddleLTBenchmark/jobs_result"
    file_path = "trimean_rope_100.png"

    upload_bos = UploadBos()
    upload_bos.upload_to_bos(bos_path, file_path)
