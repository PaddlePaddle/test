
export FLAGS_call_stack_level=2

# cd PaddleTest/distributed/CE_API && mkdir task
# cp -r run_gpu/. task && cp -r case/. task && cp -r test/. task
cd task
cases="test_dist_auto_api.py \
       test_dist_collective_communicator_api.py \
       test_dist_collective_stream_communicator_api.py \
       test_dist_countfilterentry.py \
       test_dist_rpc.py \
       test_dist_data_load.py  \
       test_dist_distattr.py \
       test_dist_dtensor_from_fn.py \
       test_dist_gloo_api.py \
       test_dist_env_training_api.py \
       test_dist_env_training_launch.py \
       test_dist_utils_api.py \
       test_dist_fleet_init.py \
       test_dist_fleet_distributedstrategy.py \
       test_dist_fleet_userdefinedrolemaker.py \
       test_dist_fleet_paddlecloudrolemaker.py \
       test_dist_fleet_utilbase.py \
       test_dist_fleet_utils.py \
       test_dist_fleet_worker.py \
       test_dist_gather.py \
       test_dist_load_state_dict.py \
       test_dist_multislotdatagenerator.py \
       test_dist_parallelmode.py \
       test_dist_partial.py \
       test_dist_placement.py \
       test_dist_probabilityentry.py \
       test_dist_processmesh.py \
       test_dist_reducetype.py \
       test_dist_replicate.py \
       test_dist_reshard.py \
       test_dist_save_state_dict.py \
       test_dist_scatter_object_list.py \
       test_dist_shard_layer.py \
       test_dist_shard_optimizer.py \
       test_dist_shard_tensor.py \
       test_dist_shard.py \
       test_dist_showclickentry.py \
       test_dist_strategy.py \
       test_dist_unshard_dtensor.py \
       test_dist_utils_recompute.py \
       test_dist_fleet_dygraph.py \
       test_dist_fleet_dygraph_loss.py \
       test_dist_fleet_hybrid_parallel.py \
       test_dist_fleet_static_launch.py \
       test_dist_fleet_static_fleetrun.py \
       test_dist_MoE_api.py \
       "
for file in ${cases}
do
    echo ${file}
    python -m pytest  ${file} --alluredir=report
done
