import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

def GetExitCodeAndStdErr(cmd, env):
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, f"last stage failed."
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

def NumOperationsInBlock(block_idx):
    return [214][block_idx] - 1 # number-of-ops-in-block

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()


if type(paddle_debug_num_allowed_ops) is not int:
    def EarlyReturn(block_idx, op_idx):
        return False      
else:
    def EarlyReturn(block_idx, op_idx):
        return op_idx >= paddle_debug_num_allowed_ops

class BlockEntries:
    def builtin_module_644_0_0(self, parameter_0, data_0, data_1):

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [22, 49, 8, 64]

        # pd_op.reshape: (22x49x8x64xf32, 0x22x49x512xi64) <- (22x49x512xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(data_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([22x49x8x16xf32, 22x49x8x16xf32, 22x49x8x32xf32]) <- (22x49x8x64xf32, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape_0, full_int_array_1, full_0)

        # builtin.split: (22x49x8x16xf32, 22x49x8x16xf32, 22x49x8x32xf32) <- ([22x49x8x16xf32, 22x49x8x16xf32, 22x49x8x32xf32])
        split_1, split_2, split_3, = split_0

        # pd_op.transpose: (22x8x49x16xf32) <- (22x49x8x16xf32)
        transpose_0 = paddle._C_ops.transpose(split_1, [0, 2, 1, 3])

        # pd_op.transpose: (22x8x49x16xf32) <- (22x49x8x16xf32)
        transpose_1 = paddle._C_ops.transpose(split_2, [0, 2, 1, 3])

        # pd_op.transpose: (22x8x49x32xf32) <- (22x49x8x32xf32)
        transpose_2 = paddle._C_ops.transpose(split_3, [0, 2, 1, 3])

        # pd_op.transpose: (22x8x16x49xf32) <- (22x8x49x16xf32)
        transpose_3 = paddle._C_ops.transpose(transpose_1, [0, 1, 3, 2])

        # pd_op.transpose: (49x8xf32) <- (8x49xf32)
        transpose_4 = paddle._C_ops.transpose(parameter_0, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_1

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(transpose_4, slice_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(transpose_4, slice_1, assign_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(data_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(transpose_4, slice_2, assign_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_1, [0], full_int_array_5, full_int_array_6, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(transpose_4, slice_3, assign_46)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [5]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(data_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(transpose_4, slice_4, assign_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [6]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(data_1, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(transpose_4, slice_5, assign_44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [7]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(data_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(transpose_4, slice_6, assign_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [8]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(data_1, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(transpose_4, slice_7, assign_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [9]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(data_1, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(transpose_4, slice_8, assign_41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [10]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(data_1, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(transpose_4, slice_9, assign_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [11]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(data_1, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(transpose_4, slice_10, assign_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [12]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(data_1, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(transpose_4, slice_11, assign_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [13]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(data_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(transpose_4, slice_12, assign_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [14]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(data_1, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(transpose_4, slice_13, assign_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [15]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(data_1, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(transpose_4, slice_14, assign_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [16]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(data_1, [0], full_int_array_17, full_int_array_18, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(transpose_4, slice_15, assign_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [17]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(data_1, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_16 = paddle._C_ops.gather(transpose_4, slice_16, assign_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [18]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(data_1, [0], full_int_array_19, full_int_array_20, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_17 = paddle._C_ops.gather(transpose_4, slice_17, assign_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [19]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(data_1, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_18 = paddle._C_ops.gather(transpose_4, slice_18, assign_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [20]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(data_1, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_19 = paddle._C_ops.gather(transpose_4, slice_19, assign_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [21]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(data_1, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_20 = paddle._C_ops.gather(transpose_4, slice_20, assign_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [22]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(data_1, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_21 = paddle._C_ops.gather(transpose_4, slice_21, assign_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [23]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(data_1, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_22 = paddle._C_ops.gather(transpose_4, slice_22, assign_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [24]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(data_1, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_23 = paddle._C_ops.gather(transpose_4, slice_23, assign_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [25]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(data_1, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_24 = paddle._C_ops.gather(transpose_4, slice_24, assign_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [26]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(data_1, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_25 = paddle._C_ops.gather(transpose_4, slice_25, assign_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [27]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(data_1, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_26 = paddle._C_ops.gather(transpose_4, slice_26, assign_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [28]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(data_1, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_27 = paddle._C_ops.gather(transpose_4, slice_27, assign_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [29]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(data_1, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_28 = paddle._C_ops.gather(transpose_4, slice_28, assign_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [30]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(data_1, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_29 = paddle._C_ops.gather(transpose_4, slice_29, assign_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [31]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(data_1, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_30 = paddle._C_ops.gather(transpose_4, slice_30, assign_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [32]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(data_1, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_31 = paddle._C_ops.gather(transpose_4, slice_31, assign_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [33]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(data_1, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_32 = paddle._C_ops.gather(transpose_4, slice_32, assign_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [34]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(data_1, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_33 = paddle._C_ops.gather(transpose_4, slice_33, assign_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [35]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(data_1, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_34 = paddle._C_ops.gather(transpose_4, slice_34, assign_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [36]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(data_1, [0], full_int_array_37, full_int_array_38, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_35 = paddle._C_ops.gather(transpose_4, slice_35, assign_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [37]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(data_1, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_36 = paddle._C_ops.gather(transpose_4, slice_36, assign_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [38]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(data_1, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_37 = paddle._C_ops.gather(transpose_4, slice_37, assign_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [39]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(data_1, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_38 = paddle._C_ops.gather(transpose_4, slice_38, assign_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [40]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(data_1, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_39 = paddle._C_ops.gather(transpose_4, slice_39, assign_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [41]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(data_1, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_40 = paddle._C_ops.gather(transpose_4, slice_40, assign_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [42]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(data_1, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_41 = paddle._C_ops.gather(transpose_4, slice_41, assign_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [43]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(data_1, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_42 = paddle._C_ops.gather(transpose_4, slice_42, assign_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [44]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(data_1, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_43 = paddle._C_ops.gather(transpose_4, slice_43, assign_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [45]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(data_1, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_44 = paddle._C_ops.gather(transpose_4, slice_44, assign_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [46]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(data_1, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_45 = paddle._C_ops.gather(transpose_4, slice_45, assign_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [47]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(data_1, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_46 = paddle._C_ops.gather(transpose_4, slice_46, assign_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [48]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(data_1, [0], full_int_array_49, full_int_array_50, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_47 = paddle._C_ops.gather(transpose_4, slice_47, assign_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [49]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(data_1, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.gather: (49x8xf32) <- (49x8xf32, 49xi64, 1xi32)
        gather_48 = paddle._C_ops.gather(transpose_4, slice_48, assign_1)

        # builtin.combine: ([49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32]) <- (49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3, gather_4, gather_5, gather_6, gather_7, gather_8, gather_9, gather_10, gather_11, gather_12, gather_13, gather_14, gather_15, gather_16, gather_17, gather_18, gather_19, gather_20, gather_21, gather_22, gather_23, gather_24, gather_25, gather_26, gather_27, gather_28, gather_29, gather_30, gather_31, gather_32, gather_33, gather_34, gather_35, gather_36, gather_37, gather_38, gather_39, gather_40, gather_41, gather_42, gather_43, gather_44, gather_45, gather_46, gather_47, gather_48]

        # pd_op.concat: (2401x8xf32) <- ([49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32, 49x8xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, assign_0)

        # pd_op.transpose: (8x2401xf32) <- (2401x8xf32)
        transpose_5 = paddle._C_ops.transpose(concat_0, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_52 = [0, 49, 49]

        # pd_op.reshape: (8x49x49xf32, 0x8x2401xi64) <- (8x2401xf32, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))
        return reshape_1, full_0, transpose_4, slice_0, full_1, gather_0, slice_1, assign_48, gather_1, slice_2, assign_47, gather_2, slice_3, assign_46, gather_3, slice_4, assign_45, gather_4, slice_5, assign_44, gather_5, slice_6, assign_43, gather_6, slice_7, assign_42, gather_7, slice_8, assign_41, gather_8, slice_9, assign_40, gather_9, slice_10, assign_39, gather_10, slice_11, assign_38, gather_11, slice_12, assign_37, gather_12, slice_13, assign_36, gather_13, slice_14, assign_35, gather_14, slice_15, assign_34, gather_15, slice_16, assign_33, gather_16, slice_17, assign_32, gather_17, slice_18, assign_31, gather_18, slice_19, assign_30, gather_19, slice_20, assign_29, gather_20, slice_21, assign_28, gather_21, slice_22, assign_27, gather_22, slice_23, assign_26, gather_23, slice_24, assign_25, gather_24, slice_25, assign_24, gather_25, slice_26, assign_23, gather_26, slice_27, assign_22, gather_27, slice_28, assign_21, gather_28, slice_29, assign_20, gather_29, slice_30, assign_19, gather_30, slice_31, assign_18, gather_31, slice_32, assign_17, gather_32, slice_33, assign_16, gather_33, slice_34, assign_15, gather_34, slice_35, assign_14, gather_35, slice_36, assign_13, gather_36, slice_37, assign_12, gather_37, slice_38, assign_11, gather_38, slice_39, assign_10, gather_39, slice_40, assign_9, gather_40, slice_41, assign_8, gather_41, slice_42, assign_7, gather_42, slice_43, assign_6, gather_43, slice_44, assign_5, gather_44, slice_45, assign_4, gather_45, slice_46, assign_3, gather_46, slice_47, assign_2, gather_47, slice_48, assign_1, gather_48, assign_0, reshape_3, transpose_0, transpose_3, reshape_2, transpose_2



def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class CinnTestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.entry(use_cinn=False)
        cinn_outs = self.entry(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y

class ModuleOp(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, data_0, data_1):
        return self.builtin_module_644_0_0(parameter_0, data_0, data_1)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_644_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([22, 49, 512], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'), 'int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[8, 49], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[22, 49, 512], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = ModuleOp()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()