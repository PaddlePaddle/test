"""
This is a python script that tests all python files in ppsci/ with doctest,
with the help of ProcessPool, testing speedup can be significantly improved.
"""
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess

def run_doctest_on_file(file_path):
    """Run doctest using subprocess on a single file."""
    print(f"Testing {file_path}")
    py_version = os.getenv('py_version', '3.10')
    result = subprocess.run([f"python{py_version}", "-m", "doctest", file_path], capture_output=True, text=True)
    return file_path, result.stdout, result.stderr

def test_all_files_in_directory(directory):
    files_to_test = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                files_to_test.append(file_path)

    with ProcessPoolExecutor(8) as executor:
        # Submit all files to the process pool
        results = executor.map(run_doctest_on_file, files_to_test)

    # for file_path, stdout, stderr in results:
    #     print(f"Results for {file_path}:\n{stdout}\n{stderr}")

if __name__ == "__main__":
    test_all_files_in_directory("../../ppsci")