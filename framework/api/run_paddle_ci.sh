home=$PWD
python -m pip install pytest
python -m pip install scipy>=1.13
export FLAGS_use_curand=1
export FLAGS_set_to_1d=0

case_dir_list=('device' 'incubate' 'linalg' 'paddlebase' 'loss' 'nn' 'optimizer' 'distribution' 'utils' 'saveload')

result_array=()
for case_dir in ${case_dir_list[@]}
do
rm -rf ${home}/$case_dir/result.txt
python multithreading_case.py $case_dir
result_array[${#result_array[@]}]=$?
wait;
done


# result
echo "=============== result ================="
for case_dir in ${case_dir_list[@]}
do
echo "[$case_dir cases result]"
cat ${home}/$case_dir/result.txt
done
for EXCODE in ${result_array[*]}
do
  if [ ${EXCODE} -ne 0 ]; then
    echo 'some case not success!'
    exit 8
  fi
done
