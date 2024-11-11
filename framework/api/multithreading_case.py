# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
parallel execute cases in paddle ci
"""
import re
import time
import queue
import threading
import os
import json
import sys

taskQueue = queue.Queue()
lock = threading.RLock()

failed_ce_case_list = []
ignore_case_dir = {
    "device": [],
    "fft": [],
    "incubate": ["test_jvp.py"],
    "linalg": ["test_norm.py"],
    "loss": [],
    "nn": [
        "test_functional_celu.py",
        "test_CELU.py",
        "test_initializer_truncated_normal_new.py",
        "test_initializer_truncated_normal.py",
        "test_cosinesimilarity.py",
    ],
    "paddlebase": [
        "test_is_grad_enabled.py",
    ],
    "optimizer": [],
    "distribution": [],
    "utils": [],
    "saveload": [],
}


def worker(fun):
    """worker"""
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()


def threadPool(threadPoolNum):
    """threadPool"""
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(target=worker, args={doFun})
        thread.daemon = True
        threadPool.append(thread)
    return threadPool


def runCETest(params):
    """runCETest"""
    path = params[0]
    case = params[1]
    print("case: %s" % case)
    val = os.system("export FLAGS_call_stack_level= && cd %s && python -m pytest %s" % (path, case))
    retry_count = 0
    final_result = ""
    while val != 0:
        val = os.system("export FLAGS_call_stack_level=2 && cd %s && python -m pytest %s" % (path, case))
        retry_count = retry_count + 1
        if retry_count > 2:
            val = 0
            final_result = "Failed"
    if final_result == "Failed":
        failed_ce_case_list.append(case)
        os.system('echo "%s" >> %s/result.txt' % (case, path))


def doFun(params):
    """doFun"""
    runCETest(params)


def main(path):
    """
    1. run case
    """
    dirs = os.listdir(path)
    case_dir = path.split("/")[-1]
    os.system('echo "============ failed cases =============" >> %s/result.txt' % path)
    ignore_case_list = ignore_case_dir[case_dir]
    pool = threadPool(13)
    for i in range(pool.__len__()):
        pool[i].start()
    for case in dirs:
        if case.startswith("test") and case.endswith("py") and case not in ignore_case_list:
            params = [path, case]
            taskQueue.put(params)
    taskQueue.join()


if __name__ == "__main__":
    case_dir = sys.argv[1]
    pwd = os.getcwd()
    path = "%s/%s" % (pwd, case_dir)
    main(path)
    os.system('echo "total bugs: %s" >> %s/result.txt' % (len(failed_ce_case_list), path))
    sys.exit(len(failed_ce_case_list))
