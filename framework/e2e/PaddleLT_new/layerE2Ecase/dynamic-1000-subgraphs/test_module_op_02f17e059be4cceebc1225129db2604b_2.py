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
    return [98][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_502_0_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8):

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_0 = [data_0, data_1, data_2]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_1 = [data_3, data_4, data_5]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_1)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_2 = [data_6, data_7, data_8]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('68'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (68xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype='int64')

        # pd_op.cast: (68xf32) <- (68xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, float('0'), True)

        # builtin.combine: ([68xf32, 68xf32]) <- (68xf32, 68xf32)
        combine_3 = [scale_1, scale_1]

        # pd_op.meshgrid: ([68x68xf32, 68x68xf32]) <- ([68xf32, 68xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (68x68xf32, 68x68xf32) <- ([68x68xf32, 68x68xf32])
        split_0, split_1, = meshgrid_0

        # builtin.combine: ([68x68xf32, 68x68xf32]) <- (68x68xf32, 68x68xf32)
        combine_4 = [split_1, split_0]

        # pd_op.stack: (68x68x2xf32) <- ([68x68xf32, 68x68xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 2]

        # pd_op.reshape: (4624x2xf32, 0x68x68x2xi64) <- (68x68x2xf32, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (4624x1xf32) <- ()
        full_5 = paddle._C_ops.full([4624, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('34'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (34xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_1, full_6, full_3, dtype='int64')

        # pd_op.cast: (34xf32) <- (34xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_7, float('0'), True)

        # builtin.combine: ([34xf32, 34xf32]) <- (34xf32, 34xf32)
        combine_5 = [scale_3, scale_3]

        # pd_op.meshgrid: ([34x34xf32, 34x34xf32]) <- ([34xf32, 34xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_5)

        # builtin.split: (34x34xf32, 34x34xf32) <- ([34x34xf32, 34x34xf32])
        split_2, split_3, = meshgrid_1

        # builtin.combine: ([34x34xf32, 34x34xf32]) <- (34x34xf32, 34x34xf32)
        combine_6 = [split_3, split_2]

        # pd_op.stack: (34x34x2xf32) <- ([34x34xf32, 34x34xf32])
        stack_1 = paddle._C_ops.stack(combine_6, -1)

        # pd_op.reshape: (1156x2xf32, 0x34x34x2xi64) <- (34x34x2xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1156x1xf32) <- ()
        full_8 = paddle._C_ops.full([1156, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('17'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (17xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_1, full_9, full_3, dtype='int64')

        # pd_op.cast: (17xf32) <- (17xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_2, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_10, float('0'), True)

        # builtin.combine: ([17xf32, 17xf32]) <- (17xf32, 17xf32)
        combine_7 = [scale_5, scale_5]

        # pd_op.meshgrid: ([17x17xf32, 17x17xf32]) <- ([17xf32, 17xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_7)

        # builtin.split: (17x17xf32, 17x17xf32) <- ([17x17xf32, 17x17xf32])
        split_4, split_5, = meshgrid_2

        # builtin.combine: ([17x17xf32, 17x17xf32]) <- (17x17xf32, 17x17xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (17x17x2xf32) <- ([17x17xf32, 17x17xf32])
        stack_2 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (289x2xf32, 0x17x17x2xi64) <- (17x17x2xf32, 2xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_2, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (289x1xf32) <- ()
        full_11 = paddle._C_ops.full([289, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([4624x2xf32, 1156x2xf32, 289x2xf32]) <- (4624x2xf32, 1156x2xf32, 289x2xf32)
        combine_9 = [reshape_0, reshape_2, reshape_4]

        # pd_op.concat: (6069x2xf32) <- ([4624x2xf32, 1156x2xf32, 289x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, full_12)

        # pd_op.cast: (6069x2xf32) <- (6069x2xf32)
        cast_3 = paddle._C_ops.cast(concat_3, paddle.float32)

        # builtin.combine: ([4624x1xf32, 1156x1xf32, 289x1xf32]) <- (4624x1xf32, 1156x1xf32, 289x1xf32)
        combine_10 = [full_5, full_8, full_11]

        # pd_op.concat: (6069x1xf32) <- ([4624x1xf32, 1156x1xf32, 289x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_10, full_12)

        # pd_op.assign: (6069x1xf32) <- (6069x1xf32)
        assign_2 = concat_4

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_1, 2, full_13)

        # builtin.split: (-1x-1x-1xf32, -1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32])
        split_6, split_7, = split_with_num_0

        # pd_op.divide: (6069x2xf32) <- (6069x2xf32, 6069x1xf32)
        divide_0 = cast_3 / concat_4

        # pd_op.add: (-1x6069x2xf32) <- (-1x-1x-1xf32, 6069x2xf32)
        add_0 = split_6 + divide_0

        # pd_op.exp: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        exp_0 = paddle._C_ops.exp(split_7)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1xf32) <- (-1x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(exp_0, full_14, float('0'), True)

        # pd_op.subtract: (-1x6069x2xf32) <- (-1x6069x2xf32, -1x-1x-1xf32)
        subtract_0 = add_0 - scale_6

        # pd_op.add: (-1x6069x2xf32) <- (-1x6069x2xf32, -1x-1x-1xf32)
        add_1 = add_0 + scale_6

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x6069x2xf32, -1x6069x2xf32]) <- (-1x6069x2xf32, -1x6069x2xf32)
        combine_11 = [subtract_0, add_1]

        # pd_op.concat: (-1x6069x4xf32) <- ([-1x6069x2xf32, -1x6069x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_15)

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(cast_0, full_3, float('0.5'), True)

        # pd_op.scale: (68xf32) <- (68xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_4, float('0'), True)

        # builtin.combine: ([68xf32, 68xf32]) <- (68xf32, 68xf32)
        combine_12 = [scale_8, scale_8]

        # pd_op.meshgrid: ([68x68xf32, 68x68xf32]) <- ([68xf32, 68xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (68x68xf32, 68x68xf32) <- ([68x68xf32, 68x68xf32])
        split_8, split_9, = meshgrid_3

        # builtin.combine: ([68x68xf32, 68x68xf32]) <- (68x68xf32, 68x68xf32)
        combine_13 = [split_9, split_8]

        # pd_op.stack: (68x68x2xf32) <- ([68x68xf32, 68x68xf32])
        stack_3 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.reshape: (4624x2xf32, 0x68x68x2xi64) <- (68x68x2xf32, 2xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_3, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_1, full_3, float('0.5'), True)

        # pd_op.scale: (34xf32) <- (34xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_7, float('0'), True)

        # builtin.combine: ([34xf32, 34xf32]) <- (34xf32, 34xf32)
        combine_14 = [scale_10, scale_10]

        # pd_op.meshgrid: ([34x34xf32, 34x34xf32]) <- ([34xf32, 34xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_14)

        # builtin.split: (34x34xf32, 34x34xf32) <- ([34x34xf32, 34x34xf32])
        split_10, split_11, = meshgrid_4

        # builtin.combine: ([34x34xf32, 34x34xf32]) <- (34x34xf32, 34x34xf32)
        combine_15 = [split_11, split_10]

        # pd_op.stack: (34x34x2xf32) <- ([34x34xf32, 34x34xf32])
        stack_4 = paddle._C_ops.stack(combine_15, -1)

        # pd_op.reshape: (1156x2xf32, 0x34x34x2xi64) <- (34x34x2xf32, 2xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_4, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cast_2, full_3, float('0.5'), True)

        # pd_op.scale: (17xf32) <- (17xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(scale_11, full_10, float('0'), True)

        # builtin.combine: ([17xf32, 17xf32]) <- (17xf32, 17xf32)
        combine_16 = [scale_12, scale_12]

        # pd_op.meshgrid: ([17x17xf32, 17x17xf32]) <- ([17xf32, 17xf32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_16)

        # builtin.split: (17x17xf32, 17x17xf32) <- ([17x17xf32, 17x17xf32])
        split_12, split_13, = meshgrid_5

        # builtin.combine: ([17x17xf32, 17x17xf32]) <- (17x17xf32, 17x17xf32)
        combine_17 = [split_13, split_12]

        # pd_op.stack: (17x17x2xf32) <- ([17x17xf32, 17x17xf32])
        stack_5 = paddle._C_ops.stack(combine_17, -1)

        # pd_op.reshape: (289x2xf32, 0x17x17x2xi64) <- (17x17x2xf32, 2xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(stack_5, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([4624x2xf32, 1156x2xf32, 289x2xf32]) <- (4624x2xf32, 1156x2xf32, 289x2xf32)
        combine_18 = [reshape_6, reshape_8, reshape_10]

        # pd_op.concat: (6069x2xf32) <- ([4624x2xf32, 1156x2xf32, 289x2xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, full_12)

        # pd_op.cast: (6069x2xf32) <- (6069x2xf32)
        cast_4 = paddle._C_ops.cast(concat_6, paddle.float32)
        return full_0, assign_1, assign_0, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_0, concat_5, concat_2, cast_4, assign_2, reshape_6, reshape_8, reshape_10



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

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8):
        return self.builtin_module_502_0_0(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_502_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([1, 4624, 80], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 1156, 80], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 289, 80], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 4624, 4], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 1156, 4], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 289, 4], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([1, 4624, 1], dtype='float32', min=0, max=0.5),
            # data_7
            paddle.uniform([1, 1156, 1], dtype='float32', min=0, max=0.5),
            # data_8
            paddle.uniform([1, 289, 1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_7
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_8
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
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