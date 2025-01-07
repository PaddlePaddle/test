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
    return [8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 8, 2, 1, 2, 33, 4358][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_if_6918_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_6918_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_6932_0_0(self, reshape__0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = reshape__0
        return assign_0
    def pd_op_if_6932_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_6900_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, assign_value_1, full_1, reshape__0, assign_value_2, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_6918_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_6918_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_6932_0_0(reshape__0)
        else:
            if_2, = self.pd_op_if_6932_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__2, assign_out__1, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_7046_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_7046_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_7060_0_0(self, reshape__0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = reshape__0
        return assign_0
    def pd_op_if_7060_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7028_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, assign_value_1, assign_value_2, assign_value_0, full_1, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7046_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7046_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7060_0_0(reshape__0)
        else:
            if_2, = self.pd_op_if_7060_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__2, assign_out__3, assign_out__0, assign_out__1, assign_out__4
    def pd_op_if_7200_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_7200_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_7214_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7214_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7182_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, full_6, assign_value_0, reshape__0, full_1, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7200_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7200_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7214_0_0(if_0)
        else:
            if_2, = self.pd_op_if_7214_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_0, assign_out__0, assign_out__4, assign_out__1, assign_out__2
    def pd_op_if_7354_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_7354_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_7368_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7368_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7336_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, assign_value_1, assign_value_0, reshape__0, full_6, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7354_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7354_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7368_0_0(if_1)
        else:
            if_2, = self.pd_op_if_7368_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__2, assign_out__0, assign_out__4, if_1, assign_out__1
    def pd_op_if_7508_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_7508_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_7522_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7522_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7490_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, full_1, assign_value_2, full_6, assign_value_1, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7508_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7508_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7522_0_0(if_1)
        else:
            if_2, = self.pd_op_if_7522_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__1, assign_out__3, if_1, assign_out__2, assign_out__0
    def pd_op_if_7662_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_7662_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_7676_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7676_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7644_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, assign_value_2, full_1, full_6, assign_value_1, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7662_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7662_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7676_0_0(if_0)
        else:
            if_2, = self.pd_op_if_7676_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__3, assign_out__1, if_0, assign_out__2, assign_out__0
    def pd_op_if_7816_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_7816_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_7830_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7830_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7798_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, reshape__0, assign_value_0, full_6, assign_value_2):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7816_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7816_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7830_0_0(if_0)
        else:
            if_2, = self.pd_op_if_7830_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, assign_out__0, if_0, assign_out__3
    def pd_op_if_7970_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_7970_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_7984_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_7984_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_7952_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, full_1, reshape__0, assign_value_1, assign_value_2, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_7970_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_7970_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_7984_0_0(if_0)
        else:
            if_2, = self.pd_op_if_7984_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__1, assign_out__4, assign_out__2, assign_out__3, assign_out__0
    def pd_op_if_8124_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_8124_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_8138_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8138_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8106_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, assign_value_0, assign_value_2, full_6, assign_value_1, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8124_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8124_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8138_0_0(if_0)
        else:
            if_2, = self.pd_op_if_8138_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__0, assign_out__3, if_0, assign_out__2, assign_out__1
    def pd_op_if_8278_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_8278_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_8292_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8292_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8260_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_1, reshape__0, full_6, assign_value_2, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8278_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8278_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8292_0_0(if_1)
        else:
            if_2, = self.pd_op_if_8292_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, assign_out__4, if_1, assign_out__3, assign_out__0
    def pd_op_if_8432_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_8432_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_8446_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8446_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8414_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, full_6, assign_value_0, reshape__0, full_1, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8432_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8432_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8446_0_0(if_1)
        else:
            if_2, = self.pd_op_if_8446_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__4, assign_out__1, assign_out__2
    def pd_op_if_8586_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_8586_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_8600_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8600_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8568_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, assign_value_0, assign_value_2, full_6, full_1, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8586_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8586_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8600_0_0(if_1)
        else:
            if_2, = self.pd_op_if_8600_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__0, assign_out__3, if_1, assign_out__1, assign_out__4
    def pd_op_if_8740_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_8740_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_8754_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8754_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8722_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, reshape__0, assign_value_1, assign_value_0, full_1, assign_value_2):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8740_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8740_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8754_0_0(if_0)
        else:
            if_2, = self.pd_op_if_8754_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__4, assign_out__2, assign_out__0, assign_out__1, assign_out__3
    def pd_op_if_8894_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_8894_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_8908_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_8908_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_8876_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, reshape__0, full_6, assign_value_0, assign_value_2):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_8894_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_8894_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_8908_0_0(if_1)
        else:
            if_2, = self.pd_op_if_8908_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, if_1, assign_out__0, assign_out__3
    def pd_op_if_9048_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9048_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_9062_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9062_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9030_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_6, reshape__0, assign_value_0, full_1, assign_value_2):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9048_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9048_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9062_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9062_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_0, assign_out__4, assign_out__0, assign_out__1, assign_out__3
    def pd_op_if_9202_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9202_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_9216_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9216_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9184_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, full_6, assign_value_0, assign_value_1, reshape__0, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9202_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9202_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9216_0_0(if_1)
        else:
            if_2, = self.pd_op_if_9216_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_9356_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9356_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_9370_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9370_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9338_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_0, full_1, full_6, assign_value_2, assign_value_1, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9356_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9356_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9370_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9370_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__0, assign_out__1, if_0, assign_out__3, assign_out__2, assign_out__4
    def pd_op_if_9510_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_9510_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_9524_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9524_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9492_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, full_6, assign_value_0, assign_value_1, reshape__0, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9510_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9510_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9524_0_0(if_1)
        else:
            if_2, = self.pd_op_if_9524_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, if_1, assign_out__0, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_9664_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9664_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_9678_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9678_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9646_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_1, full_6, reshape__0, assign_value_2, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9664_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9664_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9678_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9678_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, if_0, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_9818_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9818_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_9832_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9832_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9800_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, full_6, full_1, assign_value_2, assign_value_0, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9818_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9818_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9832_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9832_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, if_0, assign_out__1, assign_out__3, assign_out__0, assign_out__2
    def pd_op_if_9972_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_9972_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_9986_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_9986_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_9954_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, assign_value_2, assign_value_0, reshape__0, full_1, full_6):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_9972_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_9972_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_9986_0_0(if_0)
        else:
            if_2, = self.pd_op_if_9986_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__3, assign_out__0, assign_out__4, assign_out__1, if_0
    def pd_op_if_10126_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10126_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_10140_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10140_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10108_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, full_6, reshape__0, assign_value_2, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10126_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10126_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10140_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10140_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, if_1, assign_out__4, assign_out__3, assign_out__0
    def pd_op_if_10280_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10280_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_10294_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10294_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10262_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, assign_value_1, reshape__0, full_6, assign_value_0, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10280_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10280_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10294_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10294_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__2, assign_out__4, if_0, assign_out__0, assign_out__1
    def pd_op_if_10434_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10434_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_10448_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10448_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10416_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, reshape__0, assign_value_0, assign_value_2, full_6):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10434_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10434_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10448_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10448_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__4, assign_out__0, assign_out__3, if_1
    def pd_op_if_10588_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10588_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_10602_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10602_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10570_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, assign_value_0, reshape__0, assign_value_1, assign_value_2, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10588_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10588_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10602_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10602_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_1, assign_out__0, assign_out__4, assign_out__2, assign_out__3, assign_out__1
    def pd_op_if_10742_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_10742_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_10756_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10756_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10724_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, assign_value_2, reshape__0, assign_value_0, assign_value_1, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10742_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10742_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10756_0_0(if_0)
        else:
            if_2, = self.pd_op_if_10756_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__3, assign_out__4, assign_out__0, assign_out__2, assign_out__1
    def pd_op_if_10896_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_10896_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_10910_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_10910_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_10878_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, assign_value_0, reshape__0, full_6, assign_value_2):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_10896_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_10896_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_10910_0_0(if_1)
        else:
            if_2, = self.pd_op_if_10910_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__0, assign_out__4, if_1, assign_out__3
    def pd_op_if_11050_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11050_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_11064_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11064_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11032_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, full_1, reshape__0, assign_value_2, assign_value_0, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11050_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11050_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11064_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11064_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__1, assign_out__4, assign_out__3, assign_out__0, assign_out__2
    def pd_op_if_11204_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11204_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_11218_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11218_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11186_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, reshape__0, assign_value_0, assign_value_1, full_1, full_6):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11204_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11204_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11218_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11218_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__4, assign_out__0, assign_out__2, assign_out__1, if_0
    def pd_op_if_11358_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11358_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_11372_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11372_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11340_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_0, assign_value_2, full_6, assign_value_1, reshape__0, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11358_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11358_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11372_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11372_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__0, assign_out__3, if_1, assign_out__2, assign_out__4, assign_out__1
    def pd_op_if_11512_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11512_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_11526_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11526_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11494_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_6, assign_value_2, full_1, assign_value_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11512_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11512_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11526_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11526_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_0, assign_out__3, assign_out__1, assign_out__0, assign_out__4
    def pd_op_if_11666_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11666_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_11680_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11680_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11648_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_6, assign_value_2, reshape__0, assign_value_0, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11666_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11666_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11680_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11680_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, if_1, assign_out__3, assign_out__4, assign_out__0, assign_out__1
    def pd_op_if_11820_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_11820_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_11834_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11834_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11802_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, full_1, assign_value_0, assign_value_2, full_6, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11820_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11820_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11834_0_0(if_0)
        else:
            if_2, = self.pd_op_if_11834_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__1, assign_out__0, assign_out__3, if_0, assign_out__2
    def pd_op_if_11974_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_11974_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_11988_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_11988_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_11956_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, full_1, reshape__0, full_6, assign_value_0, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_11974_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_11974_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_11988_0_0(if_1)
        else:
            if_2, = self.pd_op_if_11988_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__1, assign_out__4, if_1, assign_out__0, assign_out__2
    def pd_op_if_12128_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12128_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_12142_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12142_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12110_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_2, assign_value_0, reshape__0, full_6, full_1, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12128_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12128_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12142_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12142_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__3, assign_out__0, assign_out__4, if_1, assign_out__1, assign_out__2
    def pd_op_if_12282_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_12282_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_12296_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12296_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12264_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_6, assign_value_0, assign_value_2, reshape__0, assign_value_1, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12282_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12282_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12296_0_0(if_0)
        else:
            if_2, = self.pd_op_if_12296_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, if_0, assign_out__0, assign_out__3, assign_out__4, assign_out__2, assign_out__1
    def pd_op_if_12436_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12436_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_12450_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12450_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12418_0_0(self, arange_0, feed_0, slice_2, less_than_2, reshape__0, assign_value_0, full_1, assign_value_2, full_6, assign_value_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12436_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12436_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12450_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12450_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__4, assign_out__0, assign_out__1, assign_out__3, if_1, assign_out__2
    def pd_op_if_12590_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12590_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_12604_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12604_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12572_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, full_1, full_6, assign_value_0, assign_value_2, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12590_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12590_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12604_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12604_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__1, if_1, assign_out__0, assign_out__3, assign_out__4
    def pd_op_if_12744_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12744_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_12758_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12758_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12726_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, reshape__0, assign_value_2, full_6, assign_value_1, assign_value_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12744_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12744_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12758_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12758_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__4, assign_out__3, if_1, assign_out__2, assign_out__0
    def pd_op_if_12898_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return assign_0, set_value_0
    def pd_op_if_12898_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0, full_1
    def pd_op_if_12912_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_12912_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_12880_0_0(self, arange_0, feed_0, slice_2, less_than_2, assign_value_1, assign_value_2, assign_value_0, reshape__0, full_6, full_1):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_12898_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_12898_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_12912_0_0(if_1)
        else:
            if_2, = self.pd_op_if_12912_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__2, assign_out__3, assign_out__0, assign_out__4, if_1, assign_out__1
    def pd_op_if_13052_0_0(self, slice_0, cast_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('1'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2.14748e+09'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi32]) <- (xi64, xi32)
        combine_0 = [slice_0, cast_0]

        # builtin.combine: ([xi64, 1xi32]) <- (xi64, 1xi32)
        combine_1 = [scale_0, full_1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value: (-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi64, xi32], [xi64, 1xi32], 2xi64)
        set_value_0 = paddle._C_ops.set_value(reshape__0, [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], full_int_array_0, [0, 3], [0], [], [1], [float('-inf')])

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = set_value_0
        return set_value_0, assign_0
    def pd_op_if_13052_1_0(self, full_1):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return full_1, memcpy_h2d_0
    def pd_op_if_13066_0_0(self, if_0):

        # pd_op.assign: (-1x40x6x40xf32) <- (-1x40x6x40xf32)
        assign_0 = if_0
        return assign_0
    def pd_op_if_13066_1_0(self, ):

        # pd_op.full: (1x40x6x40xf32) <- ()
        full_0 = paddle._C_ops.full([1, 40, 6, 40], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x40x6x40xf32) <- (1x40x6x40xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_while_13034_0_0(self, arange_0, feed_0, slice_2, less_than_2, full_1, assign_value_1, assign_value_2, full_6, assign_value_0, reshape__0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(full_1, full_0, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]

        # pd_op.slice: (xi64) <- (-1xi64, [xi64], [xi64])
        slice_0 = paddle._C_ops.slice(arange_0, [0], [x.reshape([]) for x in combine_0], [x.reshape([]) for x in combine_1], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(full_1, full_2, float('1'), True)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [full_1]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [scale_1]

        # pd_op.slice: (xf32) <- (-1xf32, [xi64], [xi64])
        slice_1 = paddle._C_ops.slice(feed_0, [0], [x.reshape([]) for x in combine_2], [x.reshape([]) for x in combine_3], [-1], [0])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('40'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xf32) <- (xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(scale_2, full_4, float('0.5'), True)

        # pd_op.cast: (xi32) <- (xf32)
        cast_0 = paddle._C_ops.cast(scale__0, paddle.int32)

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.less_than: (xb) <- (xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(cast_0, full_5)

        # pd_op.if: (-1x40x6x40xf32, -1x40x6x40xf32) <- (xb)
        if less_than_0:
            if_0, if_1, = self.pd_op_if_13052_0_0(slice_0, cast_0, reshape__0)
        else:
            if_0, if_1, = self.pd_op_if_13052_1_0(full_6)

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(less_than_0)

        # pd_op.if: (-1x40x6x40xf32) <- (xb)
        if logical_not_0:
            if_2, = self.pd_op_if_13066_0_0(if_0)
        else:
            if_2, = self.pd_op_if_13066_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.int32)

        # pd_op.select_input: (-1x40x6x40xf32) <- (xi32, -1x40x6x40xf32, -1x40x6x40xf32)
        select_input_0 = (if_2 if cast_1 == 0 else if_1)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(full_1, full_7, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_2, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(scale_3, memcpy_h2d_0)

        # pd_op.assign_out_: (xf32) <- (xf32, xf32)
        assign_out__0 = paddle._C_ops.assign_out_(slice_1, assign_value_0)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__1 = paddle._C_ops.assign_out_(scale_3, full_1)

        # pd_op.assign_out_: (xi32) <- (xi32, xi32)
        assign_out__2 = paddle._C_ops.assign_out_(cast_0, assign_value_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__3 = paddle._C_ops.assign_out_(slice_0, assign_value_2)

        # pd_op.assign_out_: (-1x40x6x40xf32) <- (-1x40x6x40xf32, -1x40x6x40xf32)
        assign_out__4 = paddle._C_ops.assign_out_(select_input_0, reshape__0)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__5 = paddle._C_ops.assign_out_(less_than_1, less_than_2)
        return assign_out__5, assign_out__1, assign_out__2, assign_out__3, if_0, assign_out__0, assign_out__4
    def builtin_module_6478_0_0(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_28, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, feed_1, feed_2, feed_0):

        # pd_op.conv2d: (-1x64x48x160xf32) <- (-1x3x48x160xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x48x160xf32) <- (-1x64x48x160xf32, 1x64x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (-1x64x48x160xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x48x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x48x160xf32) <- (-1x64x48x160xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x128x48x160xf32) <- (-1x64x48x160xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x48x160xf32) <- (-1x128x48x160xf32, 1x128x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.batch_norm_: (-1x128x48x160xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x48x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x48x160xf32) <- (-1x128x48x160xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x128x24x80xf32) <- (-1x128x48x160xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__1, full_int_array_2, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x128x24x80xf32, 256x128x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_13, parameter_14, parameter_15, parameter_16, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_17, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_18, parameter_19, parameter_20, parameter_21, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x128x24x80xf32, 256x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_0, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_23, parameter_24, parameter_25, parameter_26, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x24x80xf32) <- (-1x256x24x80xf32, -1x256x24x80xf32)
        add__2 = paddle._C_ops.add_(batch_norm__18, batch_norm__24)

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__3 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__3, parameter_27, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_28, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x24x80xf32) <- (-1x256x24x80xf32, 1x256x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_5, reshape_4)

        # pd_op.batch_norm_: (-1x256x24x80xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x24x80xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_29, parameter_30, parameter_31, parameter_32, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x24x80xf32) <- (-1x256x24x80xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [2, 2]

        # pd_op.pool2d: (-1x256x12x40xf32) <- (-1x256x24x80xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__4, full_int_array_4, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_1, parameter_33, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_34, parameter_35, parameter_36, parameter_37, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__5, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_39, parameter_40, parameter_41, parameter_42, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, -1x256x12x40xf32)
        add__4 = paddle._C_ops.add_(batch_norm__42, pool2d_1)

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__6 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__6, parameter_43, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__7 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_48, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_49, parameter_50, parameter_51, parameter_52, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, -1x256x12x40xf32)
        add__5 = paddle._C_ops.add_(batch_norm__54, relu__6)

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__8 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 256x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__8, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_54, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x12x40xf32) <- (-1x256x12x40xf32, 1x256x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_10, reshape_6)

        # pd_op.batch_norm_: (-1x256x12x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x12x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x12x40xf32) <- (-1x256x12x40xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [2, 1]

        # pd_op.pool2d: (-1x256x6x40xf32) <- (-1x256x12x40xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__9, full_int_array_6, [2, 1], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x256x6x40xf32, 512x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_2, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__10 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__10, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x256x6x40xf32, 512x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_2, parameter_69, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__7 = paddle._C_ops.add_(batch_norm__72, batch_norm__78)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__11 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__11, parameter_74, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__12, parameter_79, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__8 = paddle._C_ops.add_(batch_norm__90, relu__11)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__13 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__13, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__14, parameter_89, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__9 = paddle._C_ops.add_(batch_norm__102, relu__13)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__15 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__15, parameter_94, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__16 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__16, parameter_99, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__10 = paddle._C_ops.add_(batch_norm__114, relu__15)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__17 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__17, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__18, parameter_109, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__11 = paddle._C_ops.add_(batch_norm__126, relu__17)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__19 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__19, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 1x512x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_22, reshape_8)

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__12, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__20 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__20, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__21 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__21, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__13 = paddle._C_ops.add_(batch_norm__144, relu__20)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__22 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__23 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__23, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__14 = paddle._C_ops.add_(batch_norm__156, relu__22)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__24 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__24, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__25, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, -1x512x6x40xf32)
        add__15 = paddle._C_ops.add_(batch_norm__168, relu__24)

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__26 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 512x512x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__26, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x6x40xf32) <- (-1x512x6x40xf32, 1x512x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_29, reshape_10)

        # pd_op.batch_norm_: (-1x512x6x40xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x6x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x6x40xf32) <- (-1x512x6x40xf32)
        relu__27 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x512x6x40xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__27, parameter_156, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_157, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_30, reshape_12)

        # pd_op.shape: (4xi32) <- (-1x512x6x40xf32)
        shape_0 = paddle._C_ops.shape(relu__27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_0 = paddle._C_ops.full([], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_0 = [slice_0, full_0]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full_with_tensor: (-1x40xi64) <- (1xf32, 2xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_1, stack_0, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('91'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40xi64) <- (-1x40xi64, 1xf32)
        scale__0 = paddle._C_ops.scale_(full_with_tensor_0, full_2, float('0'), True)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_1 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_2 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.transpose: (-1x6x40x128xf32) <- (-1x128x6x40xf32)
        transpose_0 = paddle._C_ops.transpose(add__17, [0, 2, 3, 1])

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('6'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_2, full_3, float('0'), True)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_1 = [scale_0, full_4, full_5]

        # pd_op.reshape_: (-1x40x128xf32, 0x-1x6x40x128xf32) <- (-1x6x40x128xf32, [xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_3 = paddle._C_ops.shape(reshape__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_6 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_7 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_2 = [full_6, slice_3, full_7]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_8, stack_1, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_9 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_10 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_3 = [full_9, slice_3, full_10]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_2 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_11, stack_2, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_4 = [full_with_tensor_1, full_with_tensor_2]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_5 = [parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165]

        # pd_op.full: (xui8) <- ()
        full_12 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__0, rnn__1, rnn__2, rnn__3 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_1, combine_4, combine_5, None, full_12, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_2 = paddle._C_ops.transpose(rnn__0, [1, 0, 2])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_2, full_13, full_14, full_15]

        # pd_op.reshape_: (-1x6x40x128xf32, 0x-1x40x128xf32) <- (-1x40x128xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x6x40xf32) <- (-1x6x40x128xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 128x128x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(transpose_3, parameter_166, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_31, reshape_14)

        # pd_op.relu_: (-1x128x6x40xf32) <- (-1x128x6x40xf32)
        relu__28 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 128x128x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__28, parameter_168, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_169, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x6x40xf32) <- (-1x128x6x40xf32, 1x128x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_32, reshape_16)

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 41x128xf32)
        embedding_0 = paddle._C_ops.embedding(feed_1, parameter_170, -1, False)

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_4 = paddle._C_ops.transpose(embedding_0, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_7 = [slice_1, full_16, full_17]

        # pd_op.reshape_: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__19, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_8 = [slice_1, full_18, full_19]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_5 = paddle._C_ops.transpose(transpose_4, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_5, reshape__4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_0, full_20, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_4 = paddle._C_ops.shape(scale__1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_4, full_21, full_22, full_23]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__1, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_5 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_5, full_24, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_25 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(scale_1, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_25, cast_0, full_26, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_27 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_5, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(full_27, memcpy_h2d_0)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_28 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_0 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_1 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_2 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi32, xi64, -1x40x6x40xf32, xi64, xf32) <- (xb, -1x40x6x40xf32, xi32, xi64, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_6900 = 0
        while less_than_0:
            less_than_0, full_28, assign_value_1, full_27, reshape__6, assign_value_2, assign_value_0, = self.pd_op_while_6900_0_0(arange_0, feed_2, slice_5, less_than_0, full_28, assign_value_1, full_27, reshape__6, assign_value_2, assign_value_0)
            while_loop_counter_6900 += 1
            if while_loop_counter_6900 > kWhileLoopLimit:
                break
            
        while_0, while_1, while_2, while_3, while_4, while_5, = full_28, assign_value_1, full_27, reshape__6, assign_value_2, assign_value_0,

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_10 = [slice_4, full_29, full_30]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_3, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__0 = paddle._C_ops.softmax_(reshape__8, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_18, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_1 = paddle._C_ops.matmul(softmax__0, transpose_6, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_1, [0, 2, 1])

        # pd_op.transpose: (-1x40x512xf32) <- (-1x512x40xf32)
        transpose_8 = paddle._C_ops.transpose(transpose_7, [0, 2, 1])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_1 = paddle._C_ops.embedding(scale__0, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_6 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_7 = paddle._C_ops.shape(embedding_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_31 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_32 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_33 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_11 = [full_31, slice_7, full_32]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_3 = paddle._C_ops.stack(combine_11, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_33, stack_3, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_34 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_35 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_36 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_12 = [full_34, slice_7, full_35]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_4 = paddle._C_ops.stack(combine_12, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_36, stack_4, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_9 = paddle._C_ops.transpose(embedding_1, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_13 = [full_with_tensor_3, full_with_tensor_4]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_14 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_37 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__4, rnn__5, rnn__6, rnn__7 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_9, combine_13, combine_14, None, full_37, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_10 = paddle._C_ops.transpose(rnn__4, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_11 = paddle._C_ops.transpose(transpose_10, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_15 = [slice_6, full_38, full_39]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_16 = [slice_6, full_40, full_41]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_12 = paddle._C_ops.transpose(transpose_11, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_12, reshape_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_2, full_42, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_8 = paddle._C_ops.shape(scale__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_8, full_43, full_44, full_45]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__2, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_9 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_9, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_9, full_46, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_47 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(scale_2, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_47, cast_2, full_48, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_49 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_3 = paddle._C_ops.cast(slice_9, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(cast_3, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_1 = paddle._C_ops.less_than(full_49, memcpy_h2d_1)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_50 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_3 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_4 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_5 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi32, xi64, xf32, xi64, -1x40x6x40xf32) <- (xb, -1x40x6x40xf32, xi32, xi64, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7028 = 0
        while less_than_1:
            less_than_1, full_50, assign_value_4, assign_value_5, assign_value_3, full_49, reshape__10, = self.pd_op_while_7028_0_0(arange_1, feed_2, slice_9, less_than_1, full_50, assign_value_4, assign_value_5, assign_value_3, full_49, reshape__10)
            while_loop_counter_7028 += 1
            if while_loop_counter_7028 > kWhileLoopLimit:
                break
            
        while_6, while_7, while_8, while_9, while_10, while_11, = full_50, assign_value_4, assign_value_5, assign_value_3, full_49, reshape__10,

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_18 = [slice_8, full_51, full_52]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_11, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__1 = paddle._C_ops.softmax_(reshape__12, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_22, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_3 = paddle._C_ops.matmul(softmax__1, transpose_13, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_3, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_14, [2], full_int_array_32, full_int_array_33, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_8, [1], full_int_array_34, full_int_array_35, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_19 = [slice_10, slice_11]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_19, full_53)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_4 = paddle._C_ops.matmul(concat_0, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__20 = paddle._C_ops.add_(matmul_4, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add__20, 2, full_54)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_12 = split_with_num_0[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(slice_12)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_13 = split_with_num_0[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__0 = paddle._C_ops.multiply_(slice_13, sigmoid__0)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_5 = paddle._C_ops.matmul(multiply__0, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__21 = paddle._C_ops.add_(matmul_5, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__2 = paddle._C_ops.softmax_(add__21, -1)

        # pd_op.full: (1xi64) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(softmax__2, full_55, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_4 = paddle._C_ops.cast(argmax_0, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(scale__0, cast_4, full_int_array_36, full_int_array_37, full_int_array_38, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_2 = paddle._C_ops.embedding(set_value_with_tensor__0, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_10 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_10, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_11 = paddle._C_ops.shape(embedding_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_11, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_56 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_57 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_20 = [full_56, slice_15, full_57]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_5 = paddle._C_ops.stack(combine_20, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_5 = paddle._C_ops.full_with_tensor(full_58, stack_5, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_59 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_60 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_61 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_21 = [full_59, slice_15, full_60]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_6 = paddle._C_ops.stack(combine_21, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_6 = paddle._C_ops.full_with_tensor(full_61, stack_6, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_15 = paddle._C_ops.transpose(embedding_2, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_22 = [full_with_tensor_5, full_with_tensor_6]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_23 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_62 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__8, rnn__9, rnn__10, rnn__11 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_15, combine_22, combine_23, None, full_62, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_16 = paddle._C_ops.transpose(rnn__8, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_17 = paddle._C_ops.transpose(transpose_16, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_24 = [slice_14, full_63, full_64]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_25 = [slice_14, full_65, full_66]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_18 = paddle._C_ops.transpose(transpose_17, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_6 = paddle._C_ops.matmul(transpose_18, reshape_24, False, False)

        # pd_op.full: (1xf32) <- ()
        full_67 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_6, full_67, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_12 = paddle._C_ops.shape(scale__3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_12, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_16, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__3, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_13 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_13, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_71 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_17, full_71, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_72 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_5 = paddle._C_ops.cast(scale_3, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_72, cast_5, full_73, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_74 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_6 = paddle._C_ops.cast(slice_17, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(cast_6, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_2 = paddle._C_ops.less_than(full_74, memcpy_h2d_2)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_75 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_6 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_7 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_8 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7182 = 0
        while less_than_2:
            less_than_2, assign_value_8, full_75, assign_value_6, reshape__14, full_74, assign_value_7, = self.pd_op_while_7182_0_0(arange_2, feed_2, slice_17, less_than_2, assign_value_8, full_75, assign_value_6, reshape__14, full_74, assign_value_7)
            while_loop_counter_7182 += 1
            if while_loop_counter_7182 > kWhileLoopLimit:
                break
            
        while_12, while_13, while_14, while_15, while_16, while_17, = assign_value_8, full_75, assign_value_6, reshape__14, full_74, assign_value_7,

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_27 = [slice_16, full_76, full_77]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_13, [x.reshape([]) for x in combine_27]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__3 = paddle._C_ops.softmax_(reshape__16, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_26, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_7 = paddle._C_ops.matmul(softmax__3, transpose_19, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_7, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_20, [2], full_int_array_47, full_int_array_48, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [2]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_8, [1], full_int_array_49, full_int_array_50, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_28 = [slice_18, slice_19]

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_28, full_78)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_8 = paddle._C_ops.matmul(concat_1, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__22 = paddle._C_ops.add_(matmul_8, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__22, 2, full_79)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_20 = split_with_num_1[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(slice_20)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_21 = split_with_num_1[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__1 = paddle._C_ops.multiply_(slice_21, sigmoid__1)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_9 = paddle._C_ops.matmul(multiply__1, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__23 = paddle._C_ops.add_(matmul_9, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__4 = paddle._C_ops.softmax_(add__23, -1)

        # pd_op.full: (1xi64) <- ()
        full_80 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(softmax__4, full_80, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(argmax_1, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__0, cast_7, full_int_array_51, full_int_array_52, full_int_array_53, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_3 = paddle._C_ops.embedding(set_value_with_tensor__1, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_14 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_14, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_15 = paddle._C_ops.shape(embedding_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_15, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_81 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_82 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_83 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_29 = [full_81, slice_23, full_82]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_7 = paddle._C_ops.stack(combine_29, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_7 = paddle._C_ops.full_with_tensor(full_83, stack_7, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_84 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_85 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_86 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_30 = [full_84, slice_23, full_85]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_8 = paddle._C_ops.stack(combine_30, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_8 = paddle._C_ops.full_with_tensor(full_86, stack_8, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_21 = paddle._C_ops.transpose(embedding_3, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_31 = [full_with_tensor_7, full_with_tensor_8]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_32 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_87 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__12, rnn__13, rnn__14, rnn__15 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_21, combine_31, combine_32, None, full_87, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_22 = paddle._C_ops.transpose(rnn__12, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_23 = paddle._C_ops.transpose(transpose_22, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_33 = [slice_22, full_88, full_89]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_34 = [slice_22, full_90, full_91]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_24 = paddle._C_ops.transpose(transpose_23, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_10 = paddle._C_ops.matmul(transpose_24, reshape_28, False, False)

        # pd_op.full: (1xf32) <- ()
        full_92 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_10, full_92, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_16 = paddle._C_ops.shape(scale__4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_16, [0], full_int_array_58, full_int_array_59, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_24, full_93, full_94, full_95]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__4, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_17 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_17, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_96 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_25, full_96, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_97 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_8 = paddle._C_ops.cast(scale_4, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_98 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_97, cast_8, full_98, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_99 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_9 = paddle._C_ops.cast(slice_25, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_3 = paddle._C_ops.memcpy_h2d(cast_9, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_3 = paddle._C_ops.less_than(full_99, memcpy_h2d_3)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_100 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_9 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_10 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_11 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7336 = 0
        while less_than_3:
            less_than_3, assign_value_11, assign_value_10, assign_value_9, reshape__18, full_100, full_99, = self.pd_op_while_7336_0_0(arange_3, feed_2, slice_25, less_than_3, assign_value_11, assign_value_10, assign_value_9, reshape__18, full_100, full_99)
            while_loop_counter_7336 += 1
            if while_loop_counter_7336 > kWhileLoopLimit:
                break
            
        while_18, while_19, while_20, while_21, while_22, while_23, = assign_value_11, assign_value_10, assign_value_9, reshape__18, full_100, full_99,

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_36 = [slice_24, full_101, full_102]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_22, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__5 = paddle._C_ops.softmax_(reshape__20, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_30, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_11 = paddle._C_ops.matmul(softmax__5, transpose_25, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_11, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [3]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_26, [2], full_int_array_62, full_int_array_63, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [3]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_8, [1], full_int_array_64, full_int_array_65, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_37 = [slice_26, slice_27]

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_37, full_103)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_12 = paddle._C_ops.matmul(concat_2, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__24 = paddle._C_ops.add_(matmul_12, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__24, 2, full_104)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_28 = split_with_num_2[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(slice_28)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_29 = split_with_num_2[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__2 = paddle._C_ops.multiply_(slice_29, sigmoid__2)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_13 = paddle._C_ops.matmul(multiply__2, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__25 = paddle._C_ops.add_(matmul_13, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__6 = paddle._C_ops.softmax_(add__25, -1)

        # pd_op.full: (1xi64) <- ()
        full_105 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_2 = paddle._C_ops.argmax(softmax__6, full_105, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_10 = paddle._C_ops.cast(argmax_2, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__1, cast_10, full_int_array_66, full_int_array_67, full_int_array_68, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_4 = paddle._C_ops.embedding(set_value_with_tensor__2, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_18 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_18, [0], full_int_array_69, full_int_array_70, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_19 = paddle._C_ops.shape(embedding_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_19, [0], full_int_array_71, full_int_array_72, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_106 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_107 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_108 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_38 = [full_106, slice_31, full_107]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_9 = paddle._C_ops.stack(combine_38, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_9 = paddle._C_ops.full_with_tensor(full_108, stack_9, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_109 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_110 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_111 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_39 = [full_109, slice_31, full_110]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_10 = paddle._C_ops.stack(combine_39, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_10 = paddle._C_ops.full_with_tensor(full_111, stack_10, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_27 = paddle._C_ops.transpose(embedding_4, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_40 = [full_with_tensor_9, full_with_tensor_10]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_41 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_112 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__16, rnn__17, rnn__18, rnn__19 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_27, combine_40, combine_41, None, full_112, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_28 = paddle._C_ops.transpose(rnn__16, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_29 = paddle._C_ops.transpose(transpose_28, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_42 = [slice_30, full_113, full_114]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_43 = [slice_30, full_115, full_116]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_30 = paddle._C_ops.transpose(transpose_29, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_14 = paddle._C_ops.matmul(transpose_30, reshape_32, False, False)

        # pd_op.full: (1xf32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_14, full_117, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_20 = paddle._C_ops.shape(scale__5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_20, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_32, full_118, full_119, full_120]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__5, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_21 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(shape_21, [0], full_int_array_75, full_int_array_76, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_121 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_33, full_121, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_122 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_11 = paddle._C_ops.cast(scale_5, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_123 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_122, cast_11, full_123, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_124 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_12 = paddle._C_ops.cast(slice_33, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_4 = paddle._C_ops.memcpy_h2d(cast_12, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_4 = paddle._C_ops.less_than(full_124, memcpy_h2d_4)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_125 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_12 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_13 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_14 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, -1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7490 = 0
        while less_than_4:
            less_than_4, reshape__22, full_124, assign_value_14, full_125, assign_value_13, assign_value_12, = self.pd_op_while_7490_0_0(arange_4, feed_2, slice_33, less_than_4, reshape__22, full_124, assign_value_14, full_125, assign_value_13, assign_value_12)
            while_loop_counter_7490 += 1
            if while_loop_counter_7490 > kWhileLoopLimit:
                break
            
        while_24, while_25, while_26, while_27, while_28, while_29, = reshape__22, full_124, assign_value_14, full_125, assign_value_13, assign_value_12,

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_45 = [slice_32, full_126, full_127]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_27, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__7 = paddle._C_ops.softmax_(reshape__24, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_34, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_15 = paddle._C_ops.matmul(softmax__7, transpose_31, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_15, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [4]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_32, [2], full_int_array_77, full_int_array_78, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [4]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_8, [1], full_int_array_79, full_int_array_80, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_46 = [slice_34, slice_35]

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_46, full_128)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_16 = paddle._C_ops.matmul(concat_3, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__26 = paddle._C_ops.add_(matmul_16, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__26, 2, full_129)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_36 = split_with_num_3[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(slice_36)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_37 = split_with_num_3[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__3 = paddle._C_ops.multiply_(slice_37, sigmoid__3)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_17 = paddle._C_ops.matmul(multiply__3, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__27 = paddle._C_ops.add_(matmul_17, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__8 = paddle._C_ops.softmax_(add__27, -1)

        # pd_op.full: (1xi64) <- ()
        full_130 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_3 = paddle._C_ops.argmax(softmax__8, full_130, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_13 = paddle._C_ops.cast(argmax_3, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__2, cast_13, full_int_array_81, full_int_array_82, full_int_array_83, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_5 = paddle._C_ops.embedding(set_value_with_tensor__3, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_22 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_22, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_23 = paddle._C_ops.shape(embedding_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_23, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_131 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_132 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_133 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_47 = [full_131, slice_39, full_132]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_11 = paddle._C_ops.stack(combine_47, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_11 = paddle._C_ops.full_with_tensor(full_133, stack_11, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_134 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_135 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_136 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_48 = [full_134, slice_39, full_135]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_12 = paddle._C_ops.stack(combine_48, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_12 = paddle._C_ops.full_with_tensor(full_136, stack_12, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_33 = paddle._C_ops.transpose(embedding_5, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_49 = [full_with_tensor_11, full_with_tensor_12]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_50 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_137 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__20, rnn__21, rnn__22, rnn__23 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_33, combine_49, combine_50, None, full_137, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_34 = paddle._C_ops.transpose(rnn__20, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_35 = paddle._C_ops.transpose(transpose_34, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_51 = [slice_38, full_138, full_139]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_51]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_52 = [slice_38, full_140, full_141]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_52]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_36 = paddle._C_ops.transpose(transpose_35, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_18 = paddle._C_ops.matmul(transpose_36, reshape_36, False, False)

        # pd_op.full: (1xf32) <- ()
        full_142 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_18, full_142, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_24 = paddle._C_ops.shape(scale__6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_24, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_40, full_143, full_144, full_145]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__6, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_25 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_25, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_146 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_41, full_146, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_147 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_14 = paddle._C_ops.cast(scale_6, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_148 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_147, cast_14, full_148, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_149 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_15 = paddle._C_ops.cast(slice_41, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_5 = paddle._C_ops.memcpy_h2d(cast_15, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_5 = paddle._C_ops.less_than(full_149, memcpy_h2d_5)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_150 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_15 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_16 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_17 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, -1x40x6x40xf32, xi64, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7644 = 0
        while less_than_5:
            less_than_5, reshape__26, assign_value_17, full_149, full_150, assign_value_16, assign_value_15, = self.pd_op_while_7644_0_0(arange_5, feed_2, slice_41, less_than_5, reshape__26, assign_value_17, full_149, full_150, assign_value_16, assign_value_15)
            while_loop_counter_7644 += 1
            if while_loop_counter_7644 > kWhileLoopLimit:
                break
            
        while_30, while_31, while_32, while_33, while_34, while_35, = reshape__26, assign_value_17, full_149, full_150, assign_value_16, assign_value_15,

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_54 = [slice_40, full_151, full_152]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_33, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__9 = paddle._C_ops.softmax_(reshape__28, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_38, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_19 = paddle._C_ops.matmul(softmax__9, transpose_37, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_19, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [5]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_38, [2], full_int_array_92, full_int_array_93, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [5]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_8, [1], full_int_array_94, full_int_array_95, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_55 = [slice_42, slice_43]

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_55, full_153)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_20 = paddle._C_ops.matmul(concat_4, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__28 = paddle._C_ops.add_(matmul_20, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__28, 2, full_154)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_44 = split_with_num_4[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(slice_44)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_45 = split_with_num_4[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__4 = paddle._C_ops.multiply_(slice_45, sigmoid__4)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_21 = paddle._C_ops.matmul(multiply__4, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__29 = paddle._C_ops.add_(matmul_21, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__10 = paddle._C_ops.softmax_(add__29, -1)

        # pd_op.full: (1xi64) <- ()
        full_155 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_4 = paddle._C_ops.argmax(softmax__10, full_155, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_16 = paddle._C_ops.cast(argmax_4, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__3, cast_16, full_int_array_96, full_int_array_97, full_int_array_98, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_6 = paddle._C_ops.embedding(set_value_with_tensor__4, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_26 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_26, [0], full_int_array_99, full_int_array_100, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_27 = paddle._C_ops.shape(embedding_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_27, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_156 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_157 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_158 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_56 = [full_156, slice_47, full_157]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_13 = paddle._C_ops.stack(combine_56, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_13 = paddle._C_ops.full_with_tensor(full_158, stack_13, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_159 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_160 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_161 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_57 = [full_159, slice_47, full_160]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_14 = paddle._C_ops.stack(combine_57, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_14 = paddle._C_ops.full_with_tensor(full_161, stack_14, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_39 = paddle._C_ops.transpose(embedding_6, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_58 = [full_with_tensor_13, full_with_tensor_14]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_59 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_162 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__24, rnn__25, rnn__26, rnn__27 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_39, combine_58, combine_59, None, full_162, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_40 = paddle._C_ops.transpose(rnn__24, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_41 = paddle._C_ops.transpose(transpose_40, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_60 = [slice_46, full_163, full_164]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_60]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_61 = [slice_46, full_165, full_166]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_42 = paddle._C_ops.transpose(transpose_41, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_22 = paddle._C_ops.matmul(transpose_42, reshape_40, False, False)

        # pd_op.full: (1xf32) <- ()
        full_167 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_22, full_167, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_28 = paddle._C_ops.shape(scale__7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(shape_28, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_48, full_168, full_169, full_170]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__7, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_29 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_29, [0], full_int_array_105, full_int_array_106, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_171 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_49, full_171, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_172 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_17 = paddle._C_ops.cast(scale_7, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_173 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_6 = paddle.arange(full_172, cast_17, full_173, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_174 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_18 = paddle._C_ops.cast(slice_49, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_6 = paddle._C_ops.memcpy_h2d(cast_18, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_6 = paddle._C_ops.less_than(full_174, memcpy_h2d_6)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_175 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_18 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_19 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_20 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7798 = 0
        while less_than_6:
            less_than_6, full_174, assign_value_19, reshape__30, assign_value_18, full_175, assign_value_20, = self.pd_op_while_7798_0_0(arange_6, feed_2, slice_49, less_than_6, full_174, assign_value_19, reshape__30, assign_value_18, full_175, assign_value_20)
            while_loop_counter_7798 += 1
            if while_loop_counter_7798 > kWhileLoopLimit:
                break
            
        while_36, while_37, while_38, while_39, while_40, while_41, = full_174, assign_value_19, reshape__30, assign_value_18, full_175, assign_value_20,

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_63 = [slice_48, full_176, full_177]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_40, [x.reshape([]) for x in combine_63]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__11 = paddle._C_ops.softmax_(reshape__32, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_42, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_23 = paddle._C_ops.matmul(softmax__11, transpose_43, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_44 = paddle._C_ops.transpose(matmul_23, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [6]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_44, [2], full_int_array_107, full_int_array_108, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [6]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_8, [1], full_int_array_109, full_int_array_110, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_64 = [slice_50, slice_51]

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_64, full_178)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_24 = paddle._C_ops.matmul(concat_5, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__30 = paddle._C_ops.add_(matmul_24, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__30, 2, full_179)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_52 = split_with_num_5[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(slice_52)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_53 = split_with_num_5[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__5 = paddle._C_ops.multiply_(slice_53, sigmoid__5)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_25 = paddle._C_ops.matmul(multiply__5, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__31 = paddle._C_ops.add_(matmul_25, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__12 = paddle._C_ops.softmax_(add__31, -1)

        # pd_op.full: (1xi64) <- ()
        full_180 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_5 = paddle._C_ops.argmax(softmax__12, full_180, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_19 = paddle._C_ops.cast(argmax_5, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__4, cast_19, full_int_array_111, full_int_array_112, full_int_array_113, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_7 = paddle._C_ops.embedding(set_value_with_tensor__5, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_30 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_30, [0], full_int_array_114, full_int_array_115, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_31 = paddle._C_ops.shape(embedding_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(shape_31, [0], full_int_array_116, full_int_array_117, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_181 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_182 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_183 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_65 = [full_181, slice_55, full_182]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_15 = paddle._C_ops.stack(combine_65, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_15 = paddle._C_ops.full_with_tensor(full_183, stack_15, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_184 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_185 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_186 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_66 = [full_184, slice_55, full_185]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_16 = paddle._C_ops.stack(combine_66, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_16 = paddle._C_ops.full_with_tensor(full_186, stack_16, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_45 = paddle._C_ops.transpose(embedding_7, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_67 = [full_with_tensor_15, full_with_tensor_16]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_68 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_187 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__28, rnn__29, rnn__30, rnn__31 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_45, combine_67, combine_68, None, full_187, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_46 = paddle._C_ops.transpose(rnn__28, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_47 = paddle._C_ops.transpose(transpose_46, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_69 = [slice_54, full_188, full_189]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_70 = [slice_54, full_190, full_191]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_48 = paddle._C_ops.transpose(transpose_47, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_26 = paddle._C_ops.matmul(transpose_48, reshape_44, False, False)

        # pd_op.full: (1xf32) <- ()
        full_192 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_26, full_192, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_32 = paddle._C_ops.shape(scale__8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_32, [0], full_int_array_118, full_int_array_119, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_71 = [slice_56, full_193, full_194, full_195]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__8, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_33 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_33, [0], full_int_array_120, full_int_array_121, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_196 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_57, full_196, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_197 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_20 = paddle._C_ops.cast(scale_8, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_198 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_7 = paddle.arange(full_197, cast_20, full_198, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_199 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_21 = paddle._C_ops.cast(slice_57, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_7 = paddle._C_ops.memcpy_h2d(cast_21, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_7 = paddle._C_ops.less_than(full_199, memcpy_h2d_7)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_200 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_21 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_22 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_23 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xi64, xf32) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_7952 = 0
        while less_than_7:
            less_than_7, full_200, full_199, reshape__34, assign_value_22, assign_value_23, assign_value_21, = self.pd_op_while_7952_0_0(arange_7, feed_2, slice_57, less_than_7, full_200, full_199, reshape__34, assign_value_22, assign_value_23, assign_value_21)
            while_loop_counter_7952 += 1
            if while_loop_counter_7952 > kWhileLoopLimit:
                break
            
        while_42, while_43, while_44, while_45, while_46, while_47, = full_200, full_199, reshape__34, assign_value_22, assign_value_23, assign_value_21,

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_72 = [slice_56, full_201, full_202]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_42, [x.reshape([]) for x in combine_72]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__13 = paddle._C_ops.softmax_(reshape__36, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_46, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_27 = paddle._C_ops.matmul(softmax__13, transpose_49, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_27, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [7]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_50, [2], full_int_array_122, full_int_array_123, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [7]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_8, [1], full_int_array_124, full_int_array_125, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_73 = [slice_58, slice_59]

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_73, full_203)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_28 = paddle._C_ops.matmul(concat_6, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__32 = paddle._C_ops.add_(matmul_28, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__32, 2, full_204)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_60 = split_with_num_6[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(slice_60)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_61 = split_with_num_6[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__6 = paddle._C_ops.multiply_(slice_61, sigmoid__6)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_29 = paddle._C_ops.matmul(multiply__6, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__33 = paddle._C_ops.add_(matmul_29, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__14 = paddle._C_ops.softmax_(add__33, -1)

        # pd_op.full: (1xi64) <- ()
        full_205 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_6 = paddle._C_ops.argmax(softmax__14, full_205, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_22 = paddle._C_ops.cast(argmax_6, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__5, cast_22, full_int_array_126, full_int_array_127, full_int_array_128, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_8 = paddle._C_ops.embedding(set_value_with_tensor__6, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_34 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_34, [0], full_int_array_129, full_int_array_130, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_35 = paddle._C_ops.shape(embedding_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_35, [0], full_int_array_131, full_int_array_132, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_206 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_207 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_208 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_74 = [full_206, slice_63, full_207]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_17 = paddle._C_ops.stack(combine_74, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_17 = paddle._C_ops.full_with_tensor(full_208, stack_17, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_209 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_210 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_211 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_75 = [full_209, slice_63, full_210]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_18 = paddle._C_ops.stack(combine_75, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_18 = paddle._C_ops.full_with_tensor(full_211, stack_18, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_51 = paddle._C_ops.transpose(embedding_8, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_76 = [full_with_tensor_17, full_with_tensor_18]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_77 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_212 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__32, rnn__33, rnn__34, rnn__35 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_51, combine_76, combine_77, None, full_212, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_52 = paddle._C_ops.transpose(rnn__32, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_53 = paddle._C_ops.transpose(transpose_52, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_78 = [slice_62, full_213, full_214]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_79 = [slice_62, full_215, full_216]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_54 = paddle._C_ops.transpose(transpose_53, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_30 = paddle._C_ops.matmul(transpose_54, reshape_48, False, False)

        # pd_op.full: (1xf32) <- ()
        full_217 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_30, full_217, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_36 = paddle._C_ops.shape(scale__9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_36, [0], full_int_array_133, full_int_array_134, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_80 = [slice_64, full_218, full_219, full_220]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__9, [x.reshape([]) for x in combine_80]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_37 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_37, [0], full_int_array_135, full_int_array_136, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_221 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_65, full_221, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_222 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_23 = paddle._C_ops.cast(scale_9, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_223 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_8 = paddle.arange(full_222, cast_23, full_223, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_224 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_24 = paddle._C_ops.cast(slice_65, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_8 = paddle._C_ops.memcpy_h2d(cast_24, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_8 = paddle._C_ops.less_than(full_224, memcpy_h2d_8)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_225 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_24 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_25 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_26 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8106 = 0
        while less_than_8:
            less_than_8, reshape__38, assign_value_24, assign_value_26, full_225, assign_value_25, full_224, = self.pd_op_while_8106_0_0(arange_8, feed_2, slice_65, less_than_8, reshape__38, assign_value_24, assign_value_26, full_225, assign_value_25, full_224)
            while_loop_counter_8106 += 1
            if while_loop_counter_8106 > kWhileLoopLimit:
                break
            
        while_48, while_49, while_50, while_51, while_52, while_53, = reshape__38, assign_value_24, assign_value_26, full_225, assign_value_25, full_224,

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_81 = [slice_64, full_226, full_227]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_51, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__15 = paddle._C_ops.softmax_(reshape__40, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_50, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_31 = paddle._C_ops.matmul(softmax__15, transpose_55, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_56 = paddle._C_ops.transpose(matmul_31, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [8]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_56, [2], full_int_array_137, full_int_array_138, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [8]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(transpose_8, [1], full_int_array_139, full_int_array_140, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_82 = [slice_66, slice_67]

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_82, full_228)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_32 = paddle._C_ops.matmul(concat_7, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__34 = paddle._C_ops.add_(matmul_32, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(add__34, 2, full_229)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_68 = split_with_num_7[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(slice_68)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_69 = split_with_num_7[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__7 = paddle._C_ops.multiply_(slice_69, sigmoid__7)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_33 = paddle._C_ops.matmul(multiply__7, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__35 = paddle._C_ops.add_(matmul_33, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__16 = paddle._C_ops.softmax_(add__35, -1)

        # pd_op.full: (1xi64) <- ()
        full_230 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_7 = paddle._C_ops.argmax(softmax__16, full_230, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_25 = paddle._C_ops.cast(argmax_7, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__6, cast_25, full_int_array_141, full_int_array_142, full_int_array_143, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_9 = paddle._C_ops.embedding(set_value_with_tensor__7, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_38 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_38, [0], full_int_array_144, full_int_array_145, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_39 = paddle._C_ops.shape(embedding_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_39, [0], full_int_array_146, full_int_array_147, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_231 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_232 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_233 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_83 = [full_231, slice_71, full_232]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_19 = paddle._C_ops.stack(combine_83, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_19 = paddle._C_ops.full_with_tensor(full_233, stack_19, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_234 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_235 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_236 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_84 = [full_234, slice_71, full_235]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_20 = paddle._C_ops.stack(combine_84, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_20 = paddle._C_ops.full_with_tensor(full_236, stack_20, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_57 = paddle._C_ops.transpose(embedding_9, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_85 = [full_with_tensor_19, full_with_tensor_20]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_86 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_237 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__36, rnn__37, rnn__38, rnn__39 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_57, combine_85, combine_86, None, full_237, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_58 = paddle._C_ops.transpose(rnn__36, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_59 = paddle._C_ops.transpose(transpose_58, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_87 = [slice_70, full_238, full_239]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_87]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_88 = [slice_70, full_240, full_241]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_88]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_60 = paddle._C_ops.transpose(transpose_59, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_34 = paddle._C_ops.matmul(transpose_60, reshape_52, False, False)

        # pd_op.full: (1xf32) <- ()
        full_242 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_34, full_242, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_40 = paddle._C_ops.shape(scale__10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(shape_40, [0], full_int_array_148, full_int_array_149, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_89 = [slice_72, full_243, full_244, full_245]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__10, [x.reshape([]) for x in combine_89]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_41 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(shape_41, [0], full_int_array_150, full_int_array_151, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_246 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_73, full_246, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_247 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_26 = paddle._C_ops.cast(scale_10, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_248 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_9 = paddle.arange(full_247, cast_26, full_248, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_249 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_27 = paddle._C_ops.cast(slice_73, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_9 = paddle._C_ops.memcpy_h2d(cast_27, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_9 = paddle._C_ops.less_than(full_249, memcpy_h2d_9)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_250 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_27 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_28 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_29 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8260 = 0
        while less_than_9:
            less_than_9, assign_value_28, full_249, reshape__42, full_250, assign_value_29, assign_value_27, = self.pd_op_while_8260_0_0(arange_9, feed_2, slice_73, less_than_9, assign_value_28, full_249, reshape__42, full_250, assign_value_29, assign_value_27)
            while_loop_counter_8260 += 1
            if while_loop_counter_8260 > kWhileLoopLimit:
                break
            
        while_54, while_55, while_56, while_57, while_58, while_59, = assign_value_28, full_249, reshape__42, full_250, assign_value_29, assign_value_27,

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_252 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_90 = [slice_72, full_251, full_252]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_57, [x.reshape([]) for x in combine_90]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__17 = paddle._C_ops.softmax_(reshape__44, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_54, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_35 = paddle._C_ops.matmul(softmax__17, transpose_61, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_62 = paddle._C_ops.transpose(matmul_35, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [9]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(transpose_62, [2], full_int_array_152, full_int_array_153, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [9]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_8, [1], full_int_array_154, full_int_array_155, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_91 = [slice_74, slice_75]

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_91, full_253)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_36 = paddle._C_ops.matmul(concat_8, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__36 = paddle._C_ops.add_(matmul_36, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(add__36, 2, full_254)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_76 = split_with_num_8[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(slice_76)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_77 = split_with_num_8[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__8 = paddle._C_ops.multiply_(slice_77, sigmoid__8)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_37 = paddle._C_ops.matmul(multiply__8, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__37 = paddle._C_ops.add_(matmul_37, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__18 = paddle._C_ops.softmax_(add__37, -1)

        # pd_op.full: (1xi64) <- ()
        full_255 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_8 = paddle._C_ops.argmax(softmax__18, full_255, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_28 = paddle._C_ops.cast(argmax_8, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__8 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__7, cast_28, full_int_array_156, full_int_array_157, full_int_array_158, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_10 = paddle._C_ops.embedding(set_value_with_tensor__8, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_42 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_42, [0], full_int_array_159, full_int_array_160, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_43 = paddle._C_ops.shape(embedding_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(shape_43, [0], full_int_array_161, full_int_array_162, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_256 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_257 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_258 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_92 = [full_256, slice_79, full_257]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_21 = paddle._C_ops.stack(combine_92, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_21 = paddle._C_ops.full_with_tensor(full_258, stack_21, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_259 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_260 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_261 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_93 = [full_259, slice_79, full_260]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_22 = paddle._C_ops.stack(combine_93, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_22 = paddle._C_ops.full_with_tensor(full_261, stack_22, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_63 = paddle._C_ops.transpose(embedding_10, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_94 = [full_with_tensor_21, full_with_tensor_22]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_95 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_262 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__40, rnn__41, rnn__42, rnn__43 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_63, combine_94, combine_95, None, full_262, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_64 = paddle._C_ops.transpose(rnn__40, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_65 = paddle._C_ops.transpose(transpose_64, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_264 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_96 = [slice_78, full_263, full_264]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_96]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_97 = [slice_78, full_265, full_266]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_97]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_66 = paddle._C_ops.transpose(transpose_65, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_38 = paddle._C_ops.matmul(transpose_66, reshape_56, False, False)

        # pd_op.full: (1xf32) <- ()
        full_267 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_38, full_267, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_44 = paddle._C_ops.shape(scale__11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(shape_44, [0], full_int_array_163, full_int_array_164, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_268 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_269 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_270 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_98 = [slice_80, full_268, full_269, full_270]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__11, [x.reshape([]) for x in combine_98]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_45 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_45, [0], full_int_array_165, full_int_array_166, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_271 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_81, full_271, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_272 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_29 = paddle._C_ops.cast(scale_11, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_273 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_10 = paddle.arange(full_272, cast_29, full_273, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_274 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_30 = paddle._C_ops.cast(slice_81, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_10 = paddle._C_ops.memcpy_h2d(cast_30, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_10 = paddle._C_ops.less_than(full_274, memcpy_h2d_10)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_275 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_30 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_31 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_32 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8414 = 0
        while less_than_10:
            less_than_10, assign_value_32, full_275, assign_value_30, reshape__46, full_274, assign_value_31, = self.pd_op_while_8414_0_0(arange_10, feed_2, slice_81, less_than_10, assign_value_32, full_275, assign_value_30, reshape__46, full_274, assign_value_31)
            while_loop_counter_8414 += 1
            if while_loop_counter_8414 > kWhileLoopLimit:
                break
            
        while_60, while_61, while_62, while_63, while_64, while_65, = assign_value_32, full_275, assign_value_30, reshape__46, full_274, assign_value_31,

        # pd_op.full: (1xi32) <- ()
        full_276 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_277 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_99 = [slice_80, full_276, full_277]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_61, [x.reshape([]) for x in combine_99]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__19 = paddle._C_ops.softmax_(reshape__48, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_58, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_39 = paddle._C_ops.matmul(softmax__19, transpose_67, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_68 = paddle._C_ops.transpose(matmul_39, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [10]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(transpose_68, [2], full_int_array_167, full_int_array_168, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [10]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(transpose_8, [1], full_int_array_169, full_int_array_170, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_100 = [slice_82, slice_83]

        # pd_op.full: (1xi32) <- ()
        full_278 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_100, full_278)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_40 = paddle._C_ops.matmul(concat_9, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__38 = paddle._C_ops.add_(matmul_40, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_279 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(add__38, 2, full_279)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_84 = split_with_num_9[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__9 = paddle._C_ops.sigmoid_(slice_84)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_85 = split_with_num_9[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__9 = paddle._C_ops.multiply_(slice_85, sigmoid__9)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_41 = paddle._C_ops.matmul(multiply__9, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__39 = paddle._C_ops.add_(matmul_41, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__20 = paddle._C_ops.softmax_(add__39, -1)

        # pd_op.full: (1xi64) <- ()
        full_280 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_9 = paddle._C_ops.argmax(softmax__20, full_280, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_31 = paddle._C_ops.cast(argmax_9, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__9 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__8, cast_31, full_int_array_171, full_int_array_172, full_int_array_173, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_11 = paddle._C_ops.embedding(set_value_with_tensor__9, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_46 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_46, [0], full_int_array_174, full_int_array_175, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_47 = paddle._C_ops.shape(embedding_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(shape_47, [0], full_int_array_176, full_int_array_177, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_281 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_282 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_283 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_101 = [full_281, slice_87, full_282]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_23 = paddle._C_ops.stack(combine_101, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_23 = paddle._C_ops.full_with_tensor(full_283, stack_23, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_284 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_285 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_286 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_102 = [full_284, slice_87, full_285]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_24 = paddle._C_ops.stack(combine_102, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_24 = paddle._C_ops.full_with_tensor(full_286, stack_24, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_69 = paddle._C_ops.transpose(embedding_11, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_103 = [full_with_tensor_23, full_with_tensor_24]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_104 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_287 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__44, rnn__45, rnn__46, rnn__47 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_69, combine_103, combine_104, None, full_287, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_70 = paddle._C_ops.transpose(rnn__44, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_71 = paddle._C_ops.transpose(transpose_70, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_288 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_289 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_105 = [slice_86, full_288, full_289]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_105]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_290 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_291 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_106 = [slice_86, full_290, full_291]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_106]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_72 = paddle._C_ops.transpose(transpose_71, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_42 = paddle._C_ops.matmul(transpose_72, reshape_60, False, False)

        # pd_op.full: (1xf32) <- ()
        full_292 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_42, full_292, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_48 = paddle._C_ops.shape(scale__12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(shape_48, [0], full_int_array_178, full_int_array_179, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_293 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_294 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_295 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_107 = [slice_88, full_293, full_294, full_295]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__12, [x.reshape([]) for x in combine_107]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_49 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(shape_49, [0], full_int_array_180, full_int_array_181, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_296 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_89, full_296, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_297 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_32 = paddle._C_ops.cast(scale_12, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_298 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_11 = paddle.arange(full_297, cast_32, full_298, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_299 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_33 = paddle._C_ops.cast(slice_89, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_11 = paddle._C_ops.memcpy_h2d(cast_33, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_11 = paddle._C_ops.less_than(full_299, memcpy_h2d_11)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_300 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_33 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_34 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_35 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xf32, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xf32, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8568 = 0
        while less_than_11:
            less_than_11, assign_value_34, assign_value_33, assign_value_35, full_300, full_299, reshape__50, = self.pd_op_while_8568_0_0(arange_11, feed_2, slice_89, less_than_11, assign_value_34, assign_value_33, assign_value_35, full_300, full_299, reshape__50)
            while_loop_counter_8568 += 1
            if while_loop_counter_8568 > kWhileLoopLimit:
                break
            
        while_66, while_67, while_68, while_69, while_70, while_71, = assign_value_34, assign_value_33, assign_value_35, full_300, full_299, reshape__50,

        # pd_op.full: (1xi32) <- ()
        full_301 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_302 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_108 = [slice_88, full_301, full_302]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_69, [x.reshape([]) for x in combine_108]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__21 = paddle._C_ops.softmax_(reshape__52, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_62, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_43 = paddle._C_ops.matmul(softmax__21, transpose_73, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_74 = paddle._C_ops.transpose(matmul_43, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [11]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(transpose_74, [2], full_int_array_182, full_int_array_183, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_184 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [11]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(transpose_8, [1], full_int_array_184, full_int_array_185, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_109 = [slice_90, slice_91]

        # pd_op.full: (1xi32) <- ()
        full_303 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_109, full_303)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_44 = paddle._C_ops.matmul(concat_10, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__40 = paddle._C_ops.add_(matmul_44, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_304 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(add__40, 2, full_304)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_92 = split_with_num_10[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__10 = paddle._C_ops.sigmoid_(slice_92)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_93 = split_with_num_10[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__10 = paddle._C_ops.multiply_(slice_93, sigmoid__10)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_45 = paddle._C_ops.matmul(multiply__10, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__41 = paddle._C_ops.add_(matmul_45, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__22 = paddle._C_ops.softmax_(add__41, -1)

        # pd_op.full: (1xi64) <- ()
        full_305 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_10 = paddle._C_ops.argmax(softmax__22, full_305, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_34 = paddle._C_ops.cast(argmax_10, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__10 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__9, cast_34, full_int_array_186, full_int_array_187, full_int_array_188, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_12 = paddle._C_ops.embedding(set_value_with_tensor__10, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_50 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_50, [0], full_int_array_189, full_int_array_190, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_51 = paddle._C_ops.shape(embedding_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(shape_51, [0], full_int_array_191, full_int_array_192, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_306 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_307 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_308 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_110 = [full_306, slice_95, full_307]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_25 = paddle._C_ops.stack(combine_110, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_25 = paddle._C_ops.full_with_tensor(full_308, stack_25, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_309 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_310 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_311 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_111 = [full_309, slice_95, full_310]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_26 = paddle._C_ops.stack(combine_111, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_26 = paddle._C_ops.full_with_tensor(full_311, stack_26, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_75 = paddle._C_ops.transpose(embedding_12, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_112 = [full_with_tensor_25, full_with_tensor_26]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_113 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_312 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__48, rnn__49, rnn__50, rnn__51 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_75, combine_112, combine_113, None, full_312, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_76 = paddle._C_ops.transpose(rnn__48, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_77 = paddle._C_ops.transpose(transpose_76, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_313 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_314 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_114 = [slice_94, full_313, full_314]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_114]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_315 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_316 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_115 = [slice_94, full_315, full_316]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_115]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_78 = paddle._C_ops.transpose(transpose_77, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_46 = paddle._C_ops.matmul(transpose_78, reshape_64, False, False)

        # pd_op.full: (1xf32) <- ()
        full_317 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_46, full_317, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_52 = paddle._C_ops.shape(scale__13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(shape_52, [0], full_int_array_193, full_int_array_194, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_318 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_319 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_320 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_116 = [slice_96, full_318, full_319, full_320]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__13, [x.reshape([]) for x in combine_116]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_53 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(shape_53, [0], full_int_array_195, full_int_array_196, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_321 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_13 = paddle._C_ops.scale(slice_97, full_321, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_322 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_35 = paddle._C_ops.cast(scale_13, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_323 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_12 = paddle.arange(full_322, cast_35, full_323, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_324 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_36 = paddle._C_ops.cast(slice_97, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_12 = paddle._C_ops.memcpy_h2d(cast_36, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_12 = paddle._C_ops.less_than(full_324, memcpy_h2d_12)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_325 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_36 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_37 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_38 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, -1x40x6x40xf32, xi32, xf32, xi64, xi64) <- (xb, -1x40x6x40xf32, -1x40x6x40xf32, xi32, xf32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8722 = 0
        while less_than_12:
            less_than_12, full_325, reshape__54, assign_value_37, assign_value_36, full_324, assign_value_38, = self.pd_op_while_8722_0_0(arange_12, feed_2, slice_97, less_than_12, full_325, reshape__54, assign_value_37, assign_value_36, full_324, assign_value_38)
            while_loop_counter_8722 += 1
            if while_loop_counter_8722 > kWhileLoopLimit:
                break
            
        while_72, while_73, while_74, while_75, while_76, while_77, = full_325, reshape__54, assign_value_37, assign_value_36, full_324, assign_value_38,

        # pd_op.full: (1xi32) <- ()
        full_326 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_327 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_117 = [slice_96, full_326, full_327]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_72, [x.reshape([]) for x in combine_117]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__23 = paddle._C_ops.softmax_(reshape__56, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_66, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_47 = paddle._C_ops.matmul(softmax__23, transpose_79, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_47, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_198 = [12]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(transpose_80, [2], full_int_array_197, full_int_array_198, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [12]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(transpose_8, [1], full_int_array_199, full_int_array_200, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_118 = [slice_98, slice_99]

        # pd_op.full: (1xi32) <- ()
        full_328 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_118, full_328)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_48 = paddle._C_ops.matmul(concat_11, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__42 = paddle._C_ops.add_(matmul_48, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_329 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(add__42, 2, full_329)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_100 = split_with_num_11[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__11 = paddle._C_ops.sigmoid_(slice_100)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_101 = split_with_num_11[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__11 = paddle._C_ops.multiply_(slice_101, sigmoid__11)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_49 = paddle._C_ops.matmul(multiply__11, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__43 = paddle._C_ops.add_(matmul_49, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__24 = paddle._C_ops.softmax_(add__43, -1)

        # pd_op.full: (1xi64) <- ()
        full_330 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_11 = paddle._C_ops.argmax(softmax__24, full_330, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_37 = paddle._C_ops.cast(argmax_11, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_201 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__11 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__10, cast_37, full_int_array_201, full_int_array_202, full_int_array_203, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_13 = paddle._C_ops.embedding(set_value_with_tensor__11, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_54 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_204 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(shape_54, [0], full_int_array_204, full_int_array_205, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_55 = paddle._C_ops.shape(embedding_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(shape_55, [0], full_int_array_206, full_int_array_207, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_331 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_332 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_333 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_119 = [full_331, slice_103, full_332]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_27 = paddle._C_ops.stack(combine_119, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_27 = paddle._C_ops.full_with_tensor(full_333, stack_27, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_334 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_335 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_336 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_120 = [full_334, slice_103, full_335]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_28 = paddle._C_ops.stack(combine_120, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_28 = paddle._C_ops.full_with_tensor(full_336, stack_28, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_81 = paddle._C_ops.transpose(embedding_13, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_121 = [full_with_tensor_27, full_with_tensor_28]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_122 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_337 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__52, rnn__53, rnn__54, rnn__55 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_81, combine_121, combine_122, None, full_337, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_82 = paddle._C_ops.transpose(rnn__52, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_83 = paddle._C_ops.transpose(transpose_82, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_338 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_339 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_123 = [slice_102, full_338, full_339]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_123]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_340 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_341 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_124 = [slice_102, full_340, full_341]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_124]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_84 = paddle._C_ops.transpose(transpose_83, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_50 = paddle._C_ops.matmul(transpose_84, reshape_68, False, False)

        # pd_op.full: (1xf32) <- ()
        full_342 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_50, full_342, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_56 = paddle._C_ops.shape(scale__14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_56, [0], full_int_array_208, full_int_array_209, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_343 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_344 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_345 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_125 = [slice_104, full_343, full_344, full_345]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__14, [x.reshape([]) for x in combine_125]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_57 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_211 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(shape_57, [0], full_int_array_210, full_int_array_211, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_346 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_14 = paddle._C_ops.scale(slice_105, full_346, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_347 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_38 = paddle._C_ops.cast(scale_14, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_348 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_13 = paddle.arange(full_347, cast_38, full_348, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_349 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_39 = paddle._C_ops.cast(slice_105, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_13 = paddle._C_ops.memcpy_h2d(cast_39, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_13 = paddle._C_ops.less_than(full_349, memcpy_h2d_13)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_350 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_39 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_40 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_41 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_8876 = 0
        while less_than_13:
            less_than_13, full_349, assign_value_40, reshape__58, full_350, assign_value_39, assign_value_41, = self.pd_op_while_8876_0_0(arange_13, feed_2, slice_105, less_than_13, full_349, assign_value_40, reshape__58, full_350, assign_value_39, assign_value_41)
            while_loop_counter_8876 += 1
            if while_loop_counter_8876 > kWhileLoopLimit:
                break
            
        while_78, while_79, while_80, while_81, while_82, while_83, = full_349, assign_value_40, reshape__58, full_350, assign_value_39, assign_value_41,

        # pd_op.full: (1xi32) <- ()
        full_351 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_352 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_126 = [slice_104, full_351, full_352]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_81, [x.reshape([]) for x in combine_126]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__25 = paddle._C_ops.softmax_(reshape__60, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_70, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_51 = paddle._C_ops.matmul(softmax__25, transpose_85, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_86 = paddle._C_ops.transpose(matmul_51, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_212 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_213 = [13]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(transpose_86, [2], full_int_array_212, full_int_array_213, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_214 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_215 = [13]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(transpose_8, [1], full_int_array_214, full_int_array_215, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_127 = [slice_106, slice_107]

        # pd_op.full: (1xi32) <- ()
        full_353 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_127, full_353)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_52 = paddle._C_ops.matmul(concat_12, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__44 = paddle._C_ops.add_(matmul_52, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_354 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(add__44, 2, full_354)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_108 = split_with_num_12[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__12 = paddle._C_ops.sigmoid_(slice_108)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_109 = split_with_num_12[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__12 = paddle._C_ops.multiply_(slice_109, sigmoid__12)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_53 = paddle._C_ops.matmul(multiply__12, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__45 = paddle._C_ops.add_(matmul_53, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__26 = paddle._C_ops.softmax_(add__45, -1)

        # pd_op.full: (1xi64) <- ()
        full_355 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_12 = paddle._C_ops.argmax(softmax__26, full_355, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_40 = paddle._C_ops.cast(argmax_12, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_216 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_217 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_218 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__12 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__11, cast_40, full_int_array_216, full_int_array_217, full_int_array_218, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_14 = paddle._C_ops.embedding(set_value_with_tensor__12, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_58 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_219 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_220 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(shape_58, [0], full_int_array_219, full_int_array_220, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_59 = paddle._C_ops.shape(embedding_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_221 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_222 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(shape_59, [0], full_int_array_221, full_int_array_222, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_356 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_357 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_358 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_128 = [full_356, slice_111, full_357]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_29 = paddle._C_ops.stack(combine_128, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_29 = paddle._C_ops.full_with_tensor(full_358, stack_29, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_359 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_360 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_361 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_129 = [full_359, slice_111, full_360]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_30 = paddle._C_ops.stack(combine_129, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_30 = paddle._C_ops.full_with_tensor(full_361, stack_30, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_87 = paddle._C_ops.transpose(embedding_14, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_130 = [full_with_tensor_29, full_with_tensor_30]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_131 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_362 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__56, rnn__57, rnn__58, rnn__59 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_87, combine_130, combine_131, None, full_362, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_88 = paddle._C_ops.transpose(rnn__56, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_89 = paddle._C_ops.transpose(transpose_88, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_363 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_364 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_132 = [slice_110, full_363, full_364]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_132]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_365 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_366 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_133 = [slice_110, full_365, full_366]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_133]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_90 = paddle._C_ops.transpose(transpose_89, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_54 = paddle._C_ops.matmul(transpose_90, reshape_72, False, False)

        # pd_op.full: (1xf32) <- ()
        full_367 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_54, full_367, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_60 = paddle._C_ops.shape(scale__15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_223 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_224 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(shape_60, [0], full_int_array_223, full_int_array_224, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_368 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_369 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_370 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_134 = [slice_112, full_368, full_369, full_370]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__15, [x.reshape([]) for x in combine_134]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_61 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(shape_61, [0], full_int_array_225, full_int_array_226, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_371 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_15 = paddle._C_ops.scale(slice_113, full_371, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_372 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_41 = paddle._C_ops.cast(scale_15, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_373 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_14 = paddle.arange(full_372, cast_41, full_373, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_374 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_42 = paddle._C_ops.cast(slice_113, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_14 = paddle._C_ops.memcpy_h2d(cast_42, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_14 = paddle._C_ops.less_than(full_374, memcpy_h2d_14)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_375 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_42 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_43 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_44 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64, xi64) <- (xb, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9030 = 0
        while less_than_14:
            less_than_14, assign_value_43, full_375, reshape__62, assign_value_42, full_374, assign_value_44, = self.pd_op_while_9030_0_0(arange_14, feed_2, slice_113, less_than_14, assign_value_43, full_375, reshape__62, assign_value_42, full_374, assign_value_44)
            while_loop_counter_9030 += 1
            if while_loop_counter_9030 > kWhileLoopLimit:
                break
            
        while_84, while_85, while_86, while_87, while_88, while_89, = assign_value_43, full_375, reshape__62, assign_value_42, full_374, assign_value_44,

        # pd_op.full: (1xi32) <- ()
        full_376 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_377 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_135 = [slice_112, full_376, full_377]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_85, [x.reshape([]) for x in combine_135]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__27 = paddle._C_ops.softmax_(reshape__64, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_91 = paddle._C_ops.transpose(reshape_74, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_55 = paddle._C_ops.matmul(softmax__27, transpose_91, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_92 = paddle._C_ops.transpose(matmul_55, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_228 = [14]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(transpose_92, [2], full_int_array_227, full_int_array_228, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_229 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_230 = [14]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(transpose_8, [1], full_int_array_229, full_int_array_230, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_136 = [slice_114, slice_115]

        # pd_op.full: (1xi32) <- ()
        full_378 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_136, full_378)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_56 = paddle._C_ops.matmul(concat_13, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__46 = paddle._C_ops.add_(matmul_56, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_379 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(add__46, 2, full_379)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_116 = split_with_num_13[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__13 = paddle._C_ops.sigmoid_(slice_116)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_117 = split_with_num_13[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__13 = paddle._C_ops.multiply_(slice_117, sigmoid__13)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_57 = paddle._C_ops.matmul(multiply__13, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__47 = paddle._C_ops.add_(matmul_57, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__28 = paddle._C_ops.softmax_(add__47, -1)

        # pd_op.full: (1xi64) <- ()
        full_380 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_13 = paddle._C_ops.argmax(softmax__28, full_380, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_43 = paddle._C_ops.cast(argmax_13, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_231 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_232 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_233 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__13 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__12, cast_43, full_int_array_231, full_int_array_232, full_int_array_233, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_15 = paddle._C_ops.embedding(set_value_with_tensor__13, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_62 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_234 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_235 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(shape_62, [0], full_int_array_234, full_int_array_235, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_63 = paddle._C_ops.shape(embedding_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_236 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_237 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(shape_63, [0], full_int_array_236, full_int_array_237, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_381 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_382 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_383 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_137 = [full_381, slice_119, full_382]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_31 = paddle._C_ops.stack(combine_137, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_31 = paddle._C_ops.full_with_tensor(full_383, stack_31, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_384 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_385 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_386 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_138 = [full_384, slice_119, full_385]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_32 = paddle._C_ops.stack(combine_138, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_32 = paddle._C_ops.full_with_tensor(full_386, stack_32, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_93 = paddle._C_ops.transpose(embedding_15, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_139 = [full_with_tensor_31, full_with_tensor_32]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_140 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_387 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__60, rnn__61, rnn__62, rnn__63 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_93, combine_139, combine_140, None, full_387, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_94 = paddle._C_ops.transpose(rnn__60, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_95 = paddle._C_ops.transpose(transpose_94, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_388 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_389 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_141 = [slice_118, full_388, full_389]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_141]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_390 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_391 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_142 = [slice_118, full_390, full_391]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_142]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_96 = paddle._C_ops.transpose(transpose_95, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_58 = paddle._C_ops.matmul(transpose_96, reshape_76, False, False)

        # pd_op.full: (1xf32) <- ()
        full_392 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_58, full_392, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_64 = paddle._C_ops.shape(scale__16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_238 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_239 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(shape_64, [0], full_int_array_238, full_int_array_239, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_393 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_394 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_395 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_143 = [slice_120, full_393, full_394, full_395]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__16, [x.reshape([]) for x in combine_143]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_65 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_240 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_241 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(shape_65, [0], full_int_array_240, full_int_array_241, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_396 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_16 = paddle._C_ops.scale(slice_121, full_396, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_397 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_44 = paddle._C_ops.cast(scale_16, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_398 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_15 = paddle.arange(full_397, cast_44, full_398, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_399 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_45 = paddle._C_ops.cast(slice_121, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_15 = paddle._C_ops.memcpy_h2d(cast_45, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_15 = paddle._C_ops.less_than(full_399, memcpy_h2d_15)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_400 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_45 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_46 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_47 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9184 = 0
        while less_than_15:
            less_than_15, assign_value_47, full_400, assign_value_45, assign_value_46, reshape__66, full_399, = self.pd_op_while_9184_0_0(arange_15, feed_2, slice_121, less_than_15, assign_value_47, full_400, assign_value_45, assign_value_46, reshape__66, full_399)
            while_loop_counter_9184 += 1
            if while_loop_counter_9184 > kWhileLoopLimit:
                break
            
        while_90, while_91, while_92, while_93, while_94, while_95, = assign_value_47, full_400, assign_value_45, assign_value_46, reshape__66, full_399,

        # pd_op.full: (1xi32) <- ()
        full_401 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_402 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_144 = [slice_120, full_401, full_402]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_91, [x.reshape([]) for x in combine_144]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__29 = paddle._C_ops.softmax_(reshape__68, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_97 = paddle._C_ops.transpose(reshape_78, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_59 = paddle._C_ops.matmul(softmax__29, transpose_97, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_98 = paddle._C_ops.transpose(matmul_59, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_242 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_243 = [15]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(transpose_98, [2], full_int_array_242, full_int_array_243, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_244 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_245 = [15]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(transpose_8, [1], full_int_array_244, full_int_array_245, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_145 = [slice_122, slice_123]

        # pd_op.full: (1xi32) <- ()
        full_403 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_145, full_403)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_60 = paddle._C_ops.matmul(concat_14, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__48 = paddle._C_ops.add_(matmul_60, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_404 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(add__48, 2, full_404)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_124 = split_with_num_14[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__14 = paddle._C_ops.sigmoid_(slice_124)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_125 = split_with_num_14[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__14 = paddle._C_ops.multiply_(slice_125, sigmoid__14)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_61 = paddle._C_ops.matmul(multiply__14, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__49 = paddle._C_ops.add_(matmul_61, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__30 = paddle._C_ops.softmax_(add__49, -1)

        # pd_op.full: (1xi64) <- ()
        full_405 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_14 = paddle._C_ops.argmax(softmax__30, full_405, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_46 = paddle._C_ops.cast(argmax_14, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_246 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_247 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_248 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__14 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__13, cast_46, full_int_array_246, full_int_array_247, full_int_array_248, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_16 = paddle._C_ops.embedding(set_value_with_tensor__14, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_66 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_249 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_250 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(shape_66, [0], full_int_array_249, full_int_array_250, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_67 = paddle._C_ops.shape(embedding_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_251 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_252 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(shape_67, [0], full_int_array_251, full_int_array_252, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_406 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_407 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_408 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_146 = [full_406, slice_127, full_407]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_33 = paddle._C_ops.stack(combine_146, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_33 = paddle._C_ops.full_with_tensor(full_408, stack_33, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_409 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_410 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_411 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_147 = [full_409, slice_127, full_410]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_34 = paddle._C_ops.stack(combine_147, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_34 = paddle._C_ops.full_with_tensor(full_411, stack_34, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_99 = paddle._C_ops.transpose(embedding_16, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_148 = [full_with_tensor_33, full_with_tensor_34]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_149 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_412 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__64, rnn__65, rnn__66, rnn__67 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_99, combine_148, combine_149, None, full_412, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_100 = paddle._C_ops.transpose(rnn__64, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_101 = paddle._C_ops.transpose(transpose_100, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_413 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_414 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_150 = [slice_126, full_413, full_414]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_150]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_415 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_416 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_151 = [slice_126, full_415, full_416]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_151]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_102 = paddle._C_ops.transpose(transpose_101, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_62 = paddle._C_ops.matmul(transpose_102, reshape_80, False, False)

        # pd_op.full: (1xf32) <- ()
        full_417 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_62, full_417, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_68 = paddle._C_ops.shape(scale__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_253 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_254 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(shape_68, [0], full_int_array_253, full_int_array_254, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_418 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_419 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_420 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_152 = [slice_128, full_418, full_419, full_420]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__17, [x.reshape([]) for x in combine_152]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_69 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_255 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_256 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(shape_69, [0], full_int_array_255, full_int_array_256, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_421 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_17 = paddle._C_ops.scale(slice_129, full_421, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_422 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_47 = paddle._C_ops.cast(scale_17, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_423 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_16 = paddle.arange(full_422, cast_47, full_423, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_424 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_48 = paddle._C_ops.cast(slice_129, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_16 = paddle._C_ops.memcpy_h2d(cast_48, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_16 = paddle._C_ops.less_than(full_424, memcpy_h2d_16)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_425 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_48 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_49 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_50 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xf32, xi64, -1x40x6x40xf32, xi64, xi32, -1x40x6x40xf32) <- (xb, xf32, xi64, -1x40x6x40xf32, xi64, xi32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9338 = 0
        while less_than_16:
            less_than_16, assign_value_48, full_424, full_425, assign_value_50, assign_value_49, reshape__70, = self.pd_op_while_9338_0_0(arange_16, feed_2, slice_129, less_than_16, assign_value_48, full_424, full_425, assign_value_50, assign_value_49, reshape__70)
            while_loop_counter_9338 += 1
            if while_loop_counter_9338 > kWhileLoopLimit:
                break
            
        while_96, while_97, while_98, while_99, while_100, while_101, = assign_value_48, full_424, full_425, assign_value_50, assign_value_49, reshape__70,

        # pd_op.full: (1xi32) <- ()
        full_426 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_427 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_153 = [slice_128, full_426, full_427]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_98, [x.reshape([]) for x in combine_153]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__31 = paddle._C_ops.softmax_(reshape__72, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_103 = paddle._C_ops.transpose(reshape_82, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_63 = paddle._C_ops.matmul(softmax__31, transpose_103, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_104 = paddle._C_ops.transpose(matmul_63, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_257 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_258 = [16]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(transpose_104, [2], full_int_array_257, full_int_array_258, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_259 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_260 = [16]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(transpose_8, [1], full_int_array_259, full_int_array_260, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_154 = [slice_130, slice_131]

        # pd_op.full: (1xi32) <- ()
        full_428 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_154, full_428)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_64 = paddle._C_ops.matmul(concat_15, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__50 = paddle._C_ops.add_(matmul_64, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_429 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(add__50, 2, full_429)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_132 = split_with_num_15[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__15 = paddle._C_ops.sigmoid_(slice_132)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_133 = split_with_num_15[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__15 = paddle._C_ops.multiply_(slice_133, sigmoid__15)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_65 = paddle._C_ops.matmul(multiply__15, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__51 = paddle._C_ops.add_(matmul_65, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__32 = paddle._C_ops.softmax_(add__51, -1)

        # pd_op.full: (1xi64) <- ()
        full_430 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_15 = paddle._C_ops.argmax(softmax__32, full_430, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_49 = paddle._C_ops.cast(argmax_15, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_261 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_262 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_263 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__15 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__14, cast_49, full_int_array_261, full_int_array_262, full_int_array_263, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_17 = paddle._C_ops.embedding(set_value_with_tensor__15, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_70 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_264 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_265 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_70, [0], full_int_array_264, full_int_array_265, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_71 = paddle._C_ops.shape(embedding_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_266 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_267 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(shape_71, [0], full_int_array_266, full_int_array_267, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_431 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_432 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_433 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_155 = [full_431, slice_135, full_432]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_35 = paddle._C_ops.stack(combine_155, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_35 = paddle._C_ops.full_with_tensor(full_433, stack_35, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_434 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_435 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_436 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_156 = [full_434, slice_135, full_435]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_36 = paddle._C_ops.stack(combine_156, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_36 = paddle._C_ops.full_with_tensor(full_436, stack_36, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_105 = paddle._C_ops.transpose(embedding_17, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_157 = [full_with_tensor_35, full_with_tensor_36]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_158 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_437 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__68, rnn__69, rnn__70, rnn__71 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_105, combine_157, combine_158, None, full_437, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_106 = paddle._C_ops.transpose(rnn__68, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_107 = paddle._C_ops.transpose(transpose_106, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_438 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_439 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_159 = [slice_134, full_438, full_439]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_159]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_440 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_441 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_160 = [slice_134, full_440, full_441]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_160]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_108 = paddle._C_ops.transpose(transpose_107, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_66 = paddle._C_ops.matmul(transpose_108, reshape_84, False, False)

        # pd_op.full: (1xf32) <- ()
        full_442 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_66, full_442, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_72 = paddle._C_ops.shape(scale__18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_268 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_269 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(shape_72, [0], full_int_array_268, full_int_array_269, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_443 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_444 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_445 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_161 = [slice_136, full_443, full_444, full_445]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__18, [x.reshape([]) for x in combine_161]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_73 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_270 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_271 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(shape_73, [0], full_int_array_270, full_int_array_271, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_446 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_18 = paddle._C_ops.scale(slice_137, full_446, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_447 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_50 = paddle._C_ops.cast(scale_18, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_448 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_17 = paddle.arange(full_447, cast_50, full_448, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_449 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_51 = paddle._C_ops.cast(slice_137, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_17 = paddle._C_ops.memcpy_h2d(cast_51, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_17 = paddle._C_ops.less_than(full_449, memcpy_h2d_17)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_450 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_51 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_52 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_53 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9492 = 0
        while less_than_17:
            less_than_17, assign_value_53, full_450, assign_value_51, assign_value_52, reshape__74, full_449, = self.pd_op_while_9492_0_0(arange_17, feed_2, slice_137, less_than_17, assign_value_53, full_450, assign_value_51, assign_value_52, reshape__74, full_449)
            while_loop_counter_9492 += 1
            if while_loop_counter_9492 > kWhileLoopLimit:
                break
            
        while_102, while_103, while_104, while_105, while_106, while_107, = assign_value_53, full_450, assign_value_51, assign_value_52, reshape__74, full_449,

        # pd_op.full: (1xi32) <- ()
        full_451 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_452 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_162 = [slice_136, full_451, full_452]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_103, [x.reshape([]) for x in combine_162]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__33 = paddle._C_ops.softmax_(reshape__76, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_109 = paddle._C_ops.transpose(reshape_86, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_67 = paddle._C_ops.matmul(softmax__33, transpose_109, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_110 = paddle._C_ops.transpose(matmul_67, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_272 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_273 = [17]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(transpose_110, [2], full_int_array_272, full_int_array_273, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_274 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_275 = [17]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(transpose_8, [1], full_int_array_274, full_int_array_275, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_163 = [slice_138, slice_139]

        # pd_op.full: (1xi32) <- ()
        full_453 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_163, full_453)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_68 = paddle._C_ops.matmul(concat_16, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__52 = paddle._C_ops.add_(matmul_68, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_454 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(add__52, 2, full_454)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_140 = split_with_num_16[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__16 = paddle._C_ops.sigmoid_(slice_140)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_141 = split_with_num_16[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__16 = paddle._C_ops.multiply_(slice_141, sigmoid__16)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_69 = paddle._C_ops.matmul(multiply__16, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__53 = paddle._C_ops.add_(matmul_69, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__34 = paddle._C_ops.softmax_(add__53, -1)

        # pd_op.full: (1xi64) <- ()
        full_455 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_16 = paddle._C_ops.argmax(softmax__34, full_455, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_52 = paddle._C_ops.cast(argmax_16, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_276 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_277 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_278 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__16 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__15, cast_52, full_int_array_276, full_int_array_277, full_int_array_278, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_18 = paddle._C_ops.embedding(set_value_with_tensor__16, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_74 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_279 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_280 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(shape_74, [0], full_int_array_279, full_int_array_280, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_75 = paddle._C_ops.shape(embedding_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_281 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_282 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(shape_75, [0], full_int_array_281, full_int_array_282, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_456 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_457 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_458 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_164 = [full_456, slice_143, full_457]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_37 = paddle._C_ops.stack(combine_164, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_37 = paddle._C_ops.full_with_tensor(full_458, stack_37, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_459 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_460 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_461 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_165 = [full_459, slice_143, full_460]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_38 = paddle._C_ops.stack(combine_165, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_38 = paddle._C_ops.full_with_tensor(full_461, stack_38, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_111 = paddle._C_ops.transpose(embedding_18, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_166 = [full_with_tensor_37, full_with_tensor_38]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_167 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_462 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__72, rnn__73, rnn__74, rnn__75 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_111, combine_166, combine_167, None, full_462, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_112 = paddle._C_ops.transpose(rnn__72, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_113 = paddle._C_ops.transpose(transpose_112, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_463 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_464 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_168 = [slice_142, full_463, full_464]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_168]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_465 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_466 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_169 = [slice_142, full_465, full_466]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_169]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_114 = paddle._C_ops.transpose(transpose_113, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_70 = paddle._C_ops.matmul(transpose_114, reshape_88, False, False)

        # pd_op.full: (1xf32) <- ()
        full_467 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_70, full_467, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_76 = paddle._C_ops.shape(scale__19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_283 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_284 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(shape_76, [0], full_int_array_283, full_int_array_284, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_468 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_469 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_470 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_170 = [slice_144, full_468, full_469, full_470]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__19, [x.reshape([]) for x in combine_170]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_77 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_285 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_286 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(shape_77, [0], full_int_array_285, full_int_array_286, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_471 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_19 = paddle._C_ops.scale(slice_145, full_471, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_472 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_53 = paddle._C_ops.cast(scale_19, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_473 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_18 = paddle.arange(full_472, cast_53, full_473, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_474 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_54 = paddle._C_ops.cast(slice_145, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_18 = paddle._C_ops.memcpy_h2d(cast_54, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_18 = paddle._C_ops.less_than(full_474, memcpy_h2d_18)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_475 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_54 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_55 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_56 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9646 = 0
        while less_than_18:
            less_than_18, assign_value_55, full_474, full_475, reshape__78, assign_value_56, assign_value_54, = self.pd_op_while_9646_0_0(arange_18, feed_2, slice_145, less_than_18, assign_value_55, full_474, full_475, reshape__78, assign_value_56, assign_value_54)
            while_loop_counter_9646 += 1
            if while_loop_counter_9646 > kWhileLoopLimit:
                break
            
        while_108, while_109, while_110, while_111, while_112, while_113, = assign_value_55, full_474, full_475, reshape__78, assign_value_56, assign_value_54,

        # pd_op.full: (1xi32) <- ()
        full_476 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_477 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_171 = [slice_144, full_476, full_477]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_110, [x.reshape([]) for x in combine_171]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__35 = paddle._C_ops.softmax_(reshape__80, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_115 = paddle._C_ops.transpose(reshape_90, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_71 = paddle._C_ops.matmul(softmax__35, transpose_115, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_116 = paddle._C_ops.transpose(matmul_71, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_287 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_288 = [18]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(transpose_116, [2], full_int_array_287, full_int_array_288, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_289 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_290 = [18]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(transpose_8, [1], full_int_array_289, full_int_array_290, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_172 = [slice_146, slice_147]

        # pd_op.full: (1xi32) <- ()
        full_478 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_172, full_478)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_72 = paddle._C_ops.matmul(concat_17, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__54 = paddle._C_ops.add_(matmul_72, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_479 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(add__54, 2, full_479)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_148 = split_with_num_17[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__17 = paddle._C_ops.sigmoid_(slice_148)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_149 = split_with_num_17[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__17 = paddle._C_ops.multiply_(slice_149, sigmoid__17)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_73 = paddle._C_ops.matmul(multiply__17, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__55 = paddle._C_ops.add_(matmul_73, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__36 = paddle._C_ops.softmax_(add__55, -1)

        # pd_op.full: (1xi64) <- ()
        full_480 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_17 = paddle._C_ops.argmax(softmax__36, full_480, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_55 = paddle._C_ops.cast(argmax_17, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_291 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_292 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_293 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__17 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__16, cast_55, full_int_array_291, full_int_array_292, full_int_array_293, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_19 = paddle._C_ops.embedding(set_value_with_tensor__17, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_78 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_294 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_295 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(shape_78, [0], full_int_array_294, full_int_array_295, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_79 = paddle._C_ops.shape(embedding_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_296 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_297 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_151 = paddle._C_ops.slice(shape_79, [0], full_int_array_296, full_int_array_297, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_481 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_482 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_483 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_173 = [full_481, slice_151, full_482]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_39 = paddle._C_ops.stack(combine_173, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_39 = paddle._C_ops.full_with_tensor(full_483, stack_39, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_484 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_485 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_486 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_174 = [full_484, slice_151, full_485]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_40 = paddle._C_ops.stack(combine_174, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_40 = paddle._C_ops.full_with_tensor(full_486, stack_40, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_117 = paddle._C_ops.transpose(embedding_19, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_175 = [full_with_tensor_39, full_with_tensor_40]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_176 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_487 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__76, rnn__77, rnn__78, rnn__79 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_117, combine_175, combine_176, None, full_487, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_118 = paddle._C_ops.transpose(rnn__76, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_119 = paddle._C_ops.transpose(transpose_118, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_488 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_489 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_177 = [slice_150, full_488, full_489]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_177]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_490 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_491 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_178 = [slice_150, full_490, full_491]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_178]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_120 = paddle._C_ops.transpose(transpose_119, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_74 = paddle._C_ops.matmul(transpose_120, reshape_92, False, False)

        # pd_op.full: (1xf32) <- ()
        full_492 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_74, full_492, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_80 = paddle._C_ops.shape(scale__20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_298 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_299 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(shape_80, [0], full_int_array_298, full_int_array_299, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_493 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_494 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_495 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_179 = [slice_152, full_493, full_494, full_495]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__20, [x.reshape([]) for x in combine_179]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_81 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_300 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_301 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(shape_81, [0], full_int_array_300, full_int_array_301, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_496 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_20 = paddle._C_ops.scale(slice_153, full_496, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_497 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_56 = paddle._C_ops.cast(scale_20, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_498 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_19 = paddle.arange(full_497, cast_56, full_498, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_499 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_57 = paddle._C_ops.cast(slice_153, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_19 = paddle._C_ops.memcpy_h2d(cast_57, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_19 = paddle._C_ops.less_than(full_499, memcpy_h2d_19)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_500 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_57 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_58 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_59 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, -1x40x6x40xf32, xi64, xi64, xf32, xi32) <- (xb, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi64, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9800 = 0
        while less_than_19:
            less_than_19, reshape__82, full_500, full_499, assign_value_59, assign_value_57, assign_value_58, = self.pd_op_while_9800_0_0(arange_19, feed_2, slice_153, less_than_19, reshape__82, full_500, full_499, assign_value_59, assign_value_57, assign_value_58)
            while_loop_counter_9800 += 1
            if while_loop_counter_9800 > kWhileLoopLimit:
                break
            
        while_114, while_115, while_116, while_117, while_118, while_119, = reshape__82, full_500, full_499, assign_value_59, assign_value_57, assign_value_58,

        # pd_op.full: (1xi32) <- ()
        full_501 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_502 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_180 = [slice_152, full_501, full_502]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_115, [x.reshape([]) for x in combine_180]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__37 = paddle._C_ops.softmax_(reshape__84, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_121 = paddle._C_ops.transpose(reshape_94, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_75 = paddle._C_ops.matmul(softmax__37, transpose_121, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_122 = paddle._C_ops.transpose(matmul_75, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_302 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_303 = [19]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(transpose_122, [2], full_int_array_302, full_int_array_303, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_304 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_305 = [19]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(transpose_8, [1], full_int_array_304, full_int_array_305, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_181 = [slice_154, slice_155]

        # pd_op.full: (1xi32) <- ()
        full_503 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_181, full_503)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_76 = paddle._C_ops.matmul(concat_18, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__56 = paddle._C_ops.add_(matmul_76, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_504 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(add__56, 2, full_504)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_156 = split_with_num_18[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__18 = paddle._C_ops.sigmoid_(slice_156)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_157 = split_with_num_18[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__18 = paddle._C_ops.multiply_(slice_157, sigmoid__18)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_77 = paddle._C_ops.matmul(multiply__18, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__57 = paddle._C_ops.add_(matmul_77, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__38 = paddle._C_ops.softmax_(add__57, -1)

        # pd_op.full: (1xi64) <- ()
        full_505 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_18 = paddle._C_ops.argmax(softmax__38, full_505, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_58 = paddle._C_ops.cast(argmax_18, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_306 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_307 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_308 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__18 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__17, cast_58, full_int_array_306, full_int_array_307, full_int_array_308, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_20 = paddle._C_ops.embedding(set_value_with_tensor__18, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_82 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_309 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_310 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_158 = paddle._C_ops.slice(shape_82, [0], full_int_array_309, full_int_array_310, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_83 = paddle._C_ops.shape(embedding_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_311 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_312 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_159 = paddle._C_ops.slice(shape_83, [0], full_int_array_311, full_int_array_312, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_506 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_507 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_508 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_182 = [full_506, slice_159, full_507]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_41 = paddle._C_ops.stack(combine_182, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_41 = paddle._C_ops.full_with_tensor(full_508, stack_41, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_509 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_510 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_511 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_183 = [full_509, slice_159, full_510]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_42 = paddle._C_ops.stack(combine_183, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_42 = paddle._C_ops.full_with_tensor(full_511, stack_42, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_123 = paddle._C_ops.transpose(embedding_20, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_184 = [full_with_tensor_41, full_with_tensor_42]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_185 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_512 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__80, rnn__81, rnn__82, rnn__83 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_123, combine_184, combine_185, None, full_512, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_124 = paddle._C_ops.transpose(rnn__80, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_125 = paddle._C_ops.transpose(transpose_124, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_513 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_514 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_186 = [slice_158, full_513, full_514]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_186]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_515 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_516 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_187 = [slice_158, full_515, full_516]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_187]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_126 = paddle._C_ops.transpose(transpose_125, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_78 = paddle._C_ops.matmul(transpose_126, reshape_96, False, False)

        # pd_op.full: (1xf32) <- ()
        full_517 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_78, full_517, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_84 = paddle._C_ops.shape(scale__21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_313 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_314 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(shape_84, [0], full_int_array_313, full_int_array_314, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_518 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_519 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_520 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_188 = [slice_160, full_518, full_519, full_520]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__21, [x.reshape([]) for x in combine_188]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_85 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_315 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_316 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(shape_85, [0], full_int_array_315, full_int_array_316, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_521 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_21 = paddle._C_ops.scale(slice_161, full_521, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_522 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_59 = paddle._C_ops.cast(scale_21, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_523 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_20 = paddle.arange(full_522, cast_59, full_523, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_524 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_60 = paddle._C_ops.cast(slice_161, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_20 = paddle._C_ops.memcpy_h2d(cast_60, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_20 = paddle._C_ops.less_than(full_524, memcpy_h2d_20)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_525 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_60 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_61 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_62 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, xf32, -1x40x6x40xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xi64, xf32, -1x40x6x40xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_9954 = 0
        while less_than_20:
            less_than_20, assign_value_61, assign_value_62, assign_value_60, reshape__86, full_524, full_525, = self.pd_op_while_9954_0_0(arange_20, feed_2, slice_161, less_than_20, assign_value_61, assign_value_62, assign_value_60, reshape__86, full_524, full_525)
            while_loop_counter_9954 += 1
            if while_loop_counter_9954 > kWhileLoopLimit:
                break
            
        while_120, while_121, while_122, while_123, while_124, while_125, = assign_value_61, assign_value_62, assign_value_60, reshape__86, full_524, full_525,

        # pd_op.full: (1xi32) <- ()
        full_526 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_527 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_189 = [slice_160, full_526, full_527]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_125, [x.reshape([]) for x in combine_189]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__39 = paddle._C_ops.softmax_(reshape__88, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_127 = paddle._C_ops.transpose(reshape_98, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_79 = paddle._C_ops.matmul(softmax__39, transpose_127, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_128 = paddle._C_ops.transpose(matmul_79, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_317 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_318 = [20]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(transpose_128, [2], full_int_array_317, full_int_array_318, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_319 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_320 = [20]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(transpose_8, [1], full_int_array_319, full_int_array_320, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_190 = [slice_162, slice_163]

        # pd_op.full: (1xi32) <- ()
        full_528 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_190, full_528)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_80 = paddle._C_ops.matmul(concat_19, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__58 = paddle._C_ops.add_(matmul_80, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_529 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(add__58, 2, full_529)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_164 = split_with_num_19[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__19 = paddle._C_ops.sigmoid_(slice_164)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_165 = split_with_num_19[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__19 = paddle._C_ops.multiply_(slice_165, sigmoid__19)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_81 = paddle._C_ops.matmul(multiply__19, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__59 = paddle._C_ops.add_(matmul_81, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__40 = paddle._C_ops.softmax_(add__59, -1)

        # pd_op.full: (1xi64) <- ()
        full_530 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_19 = paddle._C_ops.argmax(softmax__40, full_530, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_61 = paddle._C_ops.cast(argmax_19, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_321 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_322 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_323 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__19 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__18, cast_61, full_int_array_321, full_int_array_322, full_int_array_323, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_21 = paddle._C_ops.embedding(set_value_with_tensor__19, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_86 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_324 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_325 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_166 = paddle._C_ops.slice(shape_86, [0], full_int_array_324, full_int_array_325, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_87 = paddle._C_ops.shape(embedding_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_326 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_327 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(shape_87, [0], full_int_array_326, full_int_array_327, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_531 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_532 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_533 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_191 = [full_531, slice_167, full_532]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_43 = paddle._C_ops.stack(combine_191, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_43 = paddle._C_ops.full_with_tensor(full_533, stack_43, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_534 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_535 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_536 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_192 = [full_534, slice_167, full_535]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_44 = paddle._C_ops.stack(combine_192, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_44 = paddle._C_ops.full_with_tensor(full_536, stack_44, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_129 = paddle._C_ops.transpose(embedding_21, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_193 = [full_with_tensor_43, full_with_tensor_44]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_194 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_537 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__84, rnn__85, rnn__86, rnn__87 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_129, combine_193, combine_194, None, full_537, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_130 = paddle._C_ops.transpose(rnn__84, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_131 = paddle._C_ops.transpose(transpose_130, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_538 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_539 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_195 = [slice_166, full_538, full_539]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_195]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_540 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_541 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_196 = [slice_166, full_540, full_541]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_196]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_132 = paddle._C_ops.transpose(transpose_131, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_82 = paddle._C_ops.matmul(transpose_132, reshape_100, False, False)

        # pd_op.full: (1xf32) <- ()
        full_542 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_82, full_542, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_88 = paddle._C_ops.shape(scale__22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_328 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_329 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(shape_88, [0], full_int_array_328, full_int_array_329, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_543 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_544 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_545 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_197 = [slice_168, full_543, full_544, full_545]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__22, [x.reshape([]) for x in combine_197]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_89 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_330 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_331 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(shape_89, [0], full_int_array_330, full_int_array_331, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_546 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_22 = paddle._C_ops.scale(slice_169, full_546, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_547 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_62 = paddle._C_ops.cast(scale_22, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_548 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_21 = paddle.arange(full_547, cast_62, full_548, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_549 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_63 = paddle._C_ops.cast(slice_169, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_21 = paddle._C_ops.memcpy_h2d(cast_63, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_21 = paddle._C_ops.less_than(full_549, memcpy_h2d_21)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_550 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_63 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_64 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_65 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10108 = 0
        while less_than_21:
            less_than_21, full_549, assign_value_64, full_550, reshape__90, assign_value_65, assign_value_63, = self.pd_op_while_10108_0_0(arange_21, feed_2, slice_169, less_than_21, full_549, assign_value_64, full_550, reshape__90, assign_value_65, assign_value_63)
            while_loop_counter_10108 += 1
            if while_loop_counter_10108 > kWhileLoopLimit:
                break
            
        while_126, while_127, while_128, while_129, while_130, while_131, = full_549, assign_value_64, full_550, reshape__90, assign_value_65, assign_value_63,

        # pd_op.full: (1xi32) <- ()
        full_551 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_552 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_198 = [slice_168, full_551, full_552]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_128, [x.reshape([]) for x in combine_198]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__41 = paddle._C_ops.softmax_(reshape__92, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_133 = paddle._C_ops.transpose(reshape_102, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_83 = paddle._C_ops.matmul(softmax__41, transpose_133, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_134 = paddle._C_ops.transpose(matmul_83, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_332 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_333 = [21]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(transpose_134, [2], full_int_array_332, full_int_array_333, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_334 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_335 = [21]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(transpose_8, [1], full_int_array_334, full_int_array_335, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_199 = [slice_170, slice_171]

        # pd_op.full: (1xi32) <- ()
        full_553 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_199, full_553)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_84 = paddle._C_ops.matmul(concat_20, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__60 = paddle._C_ops.add_(matmul_84, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_554 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(add__60, 2, full_554)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_172 = split_with_num_20[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__20 = paddle._C_ops.sigmoid_(slice_172)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_173 = split_with_num_20[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__20 = paddle._C_ops.multiply_(slice_173, sigmoid__20)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_85 = paddle._C_ops.matmul(multiply__20, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__61 = paddle._C_ops.add_(matmul_85, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__42 = paddle._C_ops.softmax_(add__61, -1)

        # pd_op.full: (1xi64) <- ()
        full_555 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_20 = paddle._C_ops.argmax(softmax__42, full_555, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_64 = paddle._C_ops.cast(argmax_20, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_336 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_337 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_338 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__20 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__19, cast_64, full_int_array_336, full_int_array_337, full_int_array_338, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_22 = paddle._C_ops.embedding(set_value_with_tensor__20, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_90 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_339 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_340 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(shape_90, [0], full_int_array_339, full_int_array_340, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_91 = paddle._C_ops.shape(embedding_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_341 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_342 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(shape_91, [0], full_int_array_341, full_int_array_342, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_556 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_557 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_558 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_200 = [full_556, slice_175, full_557]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_45 = paddle._C_ops.stack(combine_200, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_45 = paddle._C_ops.full_with_tensor(full_558, stack_45, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_559 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_560 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_561 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_201 = [full_559, slice_175, full_560]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_46 = paddle._C_ops.stack(combine_201, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_46 = paddle._C_ops.full_with_tensor(full_561, stack_46, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_135 = paddle._C_ops.transpose(embedding_22, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_202 = [full_with_tensor_45, full_with_tensor_46]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_203 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_562 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__88, rnn__89, rnn__90, rnn__91 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_135, combine_202, combine_203, None, full_562, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_136 = paddle._C_ops.transpose(rnn__88, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_137 = paddle._C_ops.transpose(transpose_136, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_563 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_564 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_204 = [slice_174, full_563, full_564]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_204]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_565 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_566 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_205 = [slice_174, full_565, full_566]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_205]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_138 = paddle._C_ops.transpose(transpose_137, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_86 = paddle._C_ops.matmul(transpose_138, reshape_104, False, False)

        # pd_op.full: (1xf32) <- ()
        full_567 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_86, full_567, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_92 = paddle._C_ops.shape(scale__23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_343 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_344 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(shape_92, [0], full_int_array_343, full_int_array_344, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_568 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_569 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_570 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_206 = [slice_176, full_568, full_569, full_570]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__23, [x.reshape([]) for x in combine_206]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_93 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_345 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_346 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(shape_93, [0], full_int_array_345, full_int_array_346, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_571 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_23 = paddle._C_ops.scale(slice_177, full_571, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_572 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_65 = paddle._C_ops.cast(scale_23, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_573 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_22 = paddle.arange(full_572, cast_65, full_573, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_574 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_66 = paddle._C_ops.cast(slice_177, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_22 = paddle._C_ops.memcpy_h2d(cast_66, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_22 = paddle._C_ops.less_than(full_574, memcpy_h2d_22)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_575 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_66 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_67 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_68 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64) <- (xb, xi64, xi32, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10262 = 0
        while less_than_22:
            less_than_22, assign_value_68, assign_value_67, reshape__94, full_575, assign_value_66, full_574, = self.pd_op_while_10262_0_0(arange_22, feed_2, slice_177, less_than_22, assign_value_68, assign_value_67, reshape__94, full_575, assign_value_66, full_574)
            while_loop_counter_10262 += 1
            if while_loop_counter_10262 > kWhileLoopLimit:
                break
            
        while_132, while_133, while_134, while_135, while_136, while_137, = assign_value_68, assign_value_67, reshape__94, full_575, assign_value_66, full_574,

        # pd_op.full: (1xi32) <- ()
        full_576 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_577 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_207 = [slice_176, full_576, full_577]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_135, [x.reshape([]) for x in combine_207]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__43 = paddle._C_ops.softmax_(reshape__96, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_139 = paddle._C_ops.transpose(reshape_106, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_87 = paddle._C_ops.matmul(softmax__43, transpose_139, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_140 = paddle._C_ops.transpose(matmul_87, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_347 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_348 = [22]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(transpose_140, [2], full_int_array_347, full_int_array_348, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_349 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_350 = [22]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_179 = paddle._C_ops.slice(transpose_8, [1], full_int_array_349, full_int_array_350, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_208 = [slice_178, slice_179]

        # pd_op.full: (1xi32) <- ()
        full_578 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_208, full_578)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_88 = paddle._C_ops.matmul(concat_21, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__62 = paddle._C_ops.add_(matmul_88, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_579 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(add__62, 2, full_579)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_180 = split_with_num_21[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__21 = paddle._C_ops.sigmoid_(slice_180)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_181 = split_with_num_21[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__21 = paddle._C_ops.multiply_(slice_181, sigmoid__21)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_89 = paddle._C_ops.matmul(multiply__21, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__63 = paddle._C_ops.add_(matmul_89, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__44 = paddle._C_ops.softmax_(add__63, -1)

        # pd_op.full: (1xi64) <- ()
        full_580 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_21 = paddle._C_ops.argmax(softmax__44, full_580, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_67 = paddle._C_ops.cast(argmax_21, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_351 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_352 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_353 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__21 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__20, cast_67, full_int_array_351, full_int_array_352, full_int_array_353, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_23 = paddle._C_ops.embedding(set_value_with_tensor__21, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_94 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_354 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_355 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_182 = paddle._C_ops.slice(shape_94, [0], full_int_array_354, full_int_array_355, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_95 = paddle._C_ops.shape(embedding_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_356 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_357 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_183 = paddle._C_ops.slice(shape_95, [0], full_int_array_356, full_int_array_357, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_581 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_582 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_583 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_209 = [full_581, slice_183, full_582]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_47 = paddle._C_ops.stack(combine_209, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_47 = paddle._C_ops.full_with_tensor(full_583, stack_47, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_584 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_585 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_586 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_210 = [full_584, slice_183, full_585]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_48 = paddle._C_ops.stack(combine_210, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_48 = paddle._C_ops.full_with_tensor(full_586, stack_48, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_141 = paddle._C_ops.transpose(embedding_23, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_211 = [full_with_tensor_47, full_with_tensor_48]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_212 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_587 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__92, rnn__93, rnn__94, rnn__95 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_141, combine_211, combine_212, None, full_587, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_142 = paddle._C_ops.transpose(rnn__92, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_143 = paddle._C_ops.transpose(transpose_142, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_588 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_589 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_213 = [slice_182, full_588, full_589]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_213]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_590 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_591 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_214 = [slice_182, full_590, full_591]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_214]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_144 = paddle._C_ops.transpose(transpose_143, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_90 = paddle._C_ops.matmul(transpose_144, reshape_108, False, False)

        # pd_op.full: (1xf32) <- ()
        full_592 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__24 = paddle._C_ops.scale_(matmul_90, full_592, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_96 = paddle._C_ops.shape(scale__24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_358 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_359 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_184 = paddle._C_ops.slice(shape_96, [0], full_int_array_358, full_int_array_359, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_593 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_594 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_595 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_215 = [slice_184, full_593, full_594, full_595]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__24, [x.reshape([]) for x in combine_215]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_97 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_360 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_361 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_185 = paddle._C_ops.slice(shape_97, [0], full_int_array_360, full_int_array_361, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_596 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_24 = paddle._C_ops.scale(slice_185, full_596, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_597 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_68 = paddle._C_ops.cast(scale_24, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_598 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_23 = paddle.arange(full_597, cast_68, full_598, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_599 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_69 = paddle._C_ops.cast(slice_185, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_23 = paddle._C_ops.memcpy_h2d(cast_69, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_23 = paddle._C_ops.less_than(full_599, memcpy_h2d_23)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_600 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_69 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_70 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_71 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32) <- (xb, xi64, xi32, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10416 = 0
        while less_than_23:
            less_than_23, full_599, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_600, = self.pd_op_while_10416_0_0(arange_23, feed_2, slice_185, less_than_23, full_599, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_600)
            while_loop_counter_10416 += 1
            if while_loop_counter_10416 > kWhileLoopLimit:
                break
            
        while_138, while_139, while_140, while_141, while_142, while_143, = full_599, assign_value_70, reshape__98, assign_value_69, assign_value_71, full_600,

        # pd_op.full: (1xi32) <- ()
        full_601 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_602 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_216 = [slice_184, full_601, full_602]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_143, [x.reshape([]) for x in combine_216]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__45 = paddle._C_ops.softmax_(reshape__100, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_145 = paddle._C_ops.transpose(reshape_110, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_91 = paddle._C_ops.matmul(softmax__45, transpose_145, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_146 = paddle._C_ops.transpose(matmul_91, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_362 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_363 = [23]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_186 = paddle._C_ops.slice(transpose_146, [2], full_int_array_362, full_int_array_363, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_364 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_365 = [23]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_187 = paddle._C_ops.slice(transpose_8, [1], full_int_array_364, full_int_array_365, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_217 = [slice_186, slice_187]

        # pd_op.full: (1xi32) <- ()
        full_603 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_217, full_603)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_92 = paddle._C_ops.matmul(concat_22, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__64 = paddle._C_ops.add_(matmul_92, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_604 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(add__64, 2, full_604)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_188 = split_with_num_22[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__22 = paddle._C_ops.sigmoid_(slice_188)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_189 = split_with_num_22[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__22 = paddle._C_ops.multiply_(slice_189, sigmoid__22)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_93 = paddle._C_ops.matmul(multiply__22, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__65 = paddle._C_ops.add_(matmul_93, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__46 = paddle._C_ops.softmax_(add__65, -1)

        # pd_op.full: (1xi64) <- ()
        full_605 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_22 = paddle._C_ops.argmax(softmax__46, full_605, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_70 = paddle._C_ops.cast(argmax_22, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_366 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_367 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_368 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__22 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__21, cast_70, full_int_array_366, full_int_array_367, full_int_array_368, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_24 = paddle._C_ops.embedding(set_value_with_tensor__22, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_98 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_369 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_370 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_190 = paddle._C_ops.slice(shape_98, [0], full_int_array_369, full_int_array_370, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_99 = paddle._C_ops.shape(embedding_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_371 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_372 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_191 = paddle._C_ops.slice(shape_99, [0], full_int_array_371, full_int_array_372, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_606 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_607 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_608 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_218 = [full_606, slice_191, full_607]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_49 = paddle._C_ops.stack(combine_218, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_49 = paddle._C_ops.full_with_tensor(full_608, stack_49, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_609 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_610 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_611 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_219 = [full_609, slice_191, full_610]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_50 = paddle._C_ops.stack(combine_219, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_50 = paddle._C_ops.full_with_tensor(full_611, stack_50, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_147 = paddle._C_ops.transpose(embedding_24, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_220 = [full_with_tensor_49, full_with_tensor_50]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_221 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_612 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__96, rnn__97, rnn__98, rnn__99 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_147, combine_220, combine_221, None, full_612, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_148 = paddle._C_ops.transpose(rnn__96, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_149 = paddle._C_ops.transpose(transpose_148, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_613 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_614 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_222 = [slice_190, full_613, full_614]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_222]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_615 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_616 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_223 = [slice_190, full_615, full_616]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_223]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_150 = paddle._C_ops.transpose(transpose_149, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_94 = paddle._C_ops.matmul(transpose_150, reshape_112, False, False)

        # pd_op.full: (1xf32) <- ()
        full_617 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__25 = paddle._C_ops.scale_(matmul_94, full_617, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_100 = paddle._C_ops.shape(scale__25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_373 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_374 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_192 = paddle._C_ops.slice(shape_100, [0], full_int_array_373, full_int_array_374, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_618 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_619 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_620 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_224 = [slice_192, full_618, full_619, full_620]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__25, [x.reshape([]) for x in combine_224]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_101 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_375 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_376 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_193 = paddle._C_ops.slice(shape_101, [0], full_int_array_375, full_int_array_376, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_621 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_25 = paddle._C_ops.scale(slice_193, full_621, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_622 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_71 = paddle._C_ops.cast(scale_25, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_623 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_24 = paddle.arange(full_622, cast_71, full_623, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_624 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_72 = paddle._C_ops.cast(slice_193, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_24 = paddle._C_ops.memcpy_h2d(cast_72, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_24 = paddle._C_ops.less_than(full_624, memcpy_h2d_24)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_625 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_72 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_73 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_74 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, -1x40x6x40xf32, xi32, xi64, xi64) <- (xb, -1x40x6x40xf32, xf32, -1x40x6x40xf32, xi32, xi64, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10570 = 0
        while less_than_24:
            less_than_24, full_625, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_624, = self.pd_op_while_10570_0_0(arange_24, feed_2, slice_193, less_than_24, full_625, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_624)
            while_loop_counter_10570 += 1
            if while_loop_counter_10570 > kWhileLoopLimit:
                break
            
        while_144, while_145, while_146, while_147, while_148, while_149, = full_625, assign_value_72, reshape__102, assign_value_73, assign_value_74, full_624,

        # pd_op.full: (1xi32) <- ()
        full_626 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_627 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_225 = [slice_192, full_626, full_627]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_144, [x.reshape([]) for x in combine_225]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__47 = paddle._C_ops.softmax_(reshape__104, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_151 = paddle._C_ops.transpose(reshape_114, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_95 = paddle._C_ops.matmul(softmax__47, transpose_151, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_152 = paddle._C_ops.transpose(matmul_95, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_377 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_378 = [24]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_194 = paddle._C_ops.slice(transpose_152, [2], full_int_array_377, full_int_array_378, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_379 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_380 = [24]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_195 = paddle._C_ops.slice(transpose_8, [1], full_int_array_379, full_int_array_380, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_226 = [slice_194, slice_195]

        # pd_op.full: (1xi32) <- ()
        full_628 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_226, full_628)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_96 = paddle._C_ops.matmul(concat_23, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__66 = paddle._C_ops.add_(matmul_96, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_629 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(add__66, 2, full_629)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_196 = split_with_num_23[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__23 = paddle._C_ops.sigmoid_(slice_196)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_197 = split_with_num_23[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__23 = paddle._C_ops.multiply_(slice_197, sigmoid__23)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_97 = paddle._C_ops.matmul(multiply__23, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__67 = paddle._C_ops.add_(matmul_97, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__48 = paddle._C_ops.softmax_(add__67, -1)

        # pd_op.full: (1xi64) <- ()
        full_630 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_23 = paddle._C_ops.argmax(softmax__48, full_630, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_73 = paddle._C_ops.cast(argmax_23, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_381 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_382 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_383 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__23 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__22, cast_73, full_int_array_381, full_int_array_382, full_int_array_383, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_25 = paddle._C_ops.embedding(set_value_with_tensor__23, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_102 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_384 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_385 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_198 = paddle._C_ops.slice(shape_102, [0], full_int_array_384, full_int_array_385, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_103 = paddle._C_ops.shape(embedding_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_386 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_387 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_199 = paddle._C_ops.slice(shape_103, [0], full_int_array_386, full_int_array_387, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_631 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_632 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_633 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_227 = [full_631, slice_199, full_632]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_51 = paddle._C_ops.stack(combine_227, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_51 = paddle._C_ops.full_with_tensor(full_633, stack_51, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_634 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_635 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_636 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_228 = [full_634, slice_199, full_635]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_52 = paddle._C_ops.stack(combine_228, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_52 = paddle._C_ops.full_with_tensor(full_636, stack_52, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_153 = paddle._C_ops.transpose(embedding_25, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_229 = [full_with_tensor_51, full_with_tensor_52]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_230 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_637 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__100, rnn__101, rnn__102, rnn__103 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_153, combine_229, combine_230, None, full_637, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_154 = paddle._C_ops.transpose(rnn__100, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_155 = paddle._C_ops.transpose(transpose_154, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_638 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_639 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_231 = [slice_198, full_638, full_639]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_231]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_640 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_641 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_232 = [slice_198, full_640, full_641]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_232]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_156 = paddle._C_ops.transpose(transpose_155, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_98 = paddle._C_ops.matmul(transpose_156, reshape_116, False, False)

        # pd_op.full: (1xf32) <- ()
        full_642 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__26 = paddle._C_ops.scale_(matmul_98, full_642, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_104 = paddle._C_ops.shape(scale__26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_388 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_389 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_200 = paddle._C_ops.slice(shape_104, [0], full_int_array_388, full_int_array_389, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_643 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_644 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_645 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_233 = [slice_200, full_643, full_644, full_645]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__26, [x.reshape([]) for x in combine_233]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_105 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_390 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_391 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_201 = paddle._C_ops.slice(shape_105, [0], full_int_array_390, full_int_array_391, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_646 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_26 = paddle._C_ops.scale(slice_201, full_646, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_647 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_74 = paddle._C_ops.cast(scale_26, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_648 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_25 = paddle.arange(full_647, cast_74, full_648, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_649 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_75 = paddle._C_ops.cast(slice_201, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_25 = paddle._C_ops.memcpy_h2d(cast_75, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_25 = paddle._C_ops.less_than(full_649, memcpy_h2d_25)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_650 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_75 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_76 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_77 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10724 = 0
        while less_than_25:
            less_than_25, full_650, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_649, = self.pd_op_while_10724_0_0(arange_25, feed_2, slice_201, less_than_25, full_650, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_649)
            while_loop_counter_10724 += 1
            if while_loop_counter_10724 > kWhileLoopLimit:
                break
            
        while_150, while_151, while_152, while_153, while_154, while_155, = full_650, assign_value_77, reshape__106, assign_value_75, assign_value_76, full_649,

        # pd_op.full: (1xi32) <- ()
        full_651 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_652 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_234 = [slice_200, full_651, full_652]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_150, [x.reshape([]) for x in combine_234]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__49 = paddle._C_ops.softmax_(reshape__108, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_157 = paddle._C_ops.transpose(reshape_118, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_99 = paddle._C_ops.matmul(softmax__49, transpose_157, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_158 = paddle._C_ops.transpose(matmul_99, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_392 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_393 = [25]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_202 = paddle._C_ops.slice(transpose_158, [2], full_int_array_392, full_int_array_393, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_394 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_395 = [25]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_203 = paddle._C_ops.slice(transpose_8, [1], full_int_array_394, full_int_array_395, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_235 = [slice_202, slice_203]

        # pd_op.full: (1xi32) <- ()
        full_653 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_235, full_653)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_100 = paddle._C_ops.matmul(concat_24, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__68 = paddle._C_ops.add_(matmul_100, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_654 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(add__68, 2, full_654)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_204 = split_with_num_24[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__24 = paddle._C_ops.sigmoid_(slice_204)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_205 = split_with_num_24[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__24 = paddle._C_ops.multiply_(slice_205, sigmoid__24)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_101 = paddle._C_ops.matmul(multiply__24, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__69 = paddle._C_ops.add_(matmul_101, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__50 = paddle._C_ops.softmax_(add__69, -1)

        # pd_op.full: (1xi64) <- ()
        full_655 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_24 = paddle._C_ops.argmax(softmax__50, full_655, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_76 = paddle._C_ops.cast(argmax_24, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_396 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_397 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_398 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__24 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__23, cast_76, full_int_array_396, full_int_array_397, full_int_array_398, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_26 = paddle._C_ops.embedding(set_value_with_tensor__24, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_106 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_399 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_400 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_206 = paddle._C_ops.slice(shape_106, [0], full_int_array_399, full_int_array_400, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_107 = paddle._C_ops.shape(embedding_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_401 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_402 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_207 = paddle._C_ops.slice(shape_107, [0], full_int_array_401, full_int_array_402, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_656 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_657 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_658 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_236 = [full_656, slice_207, full_657]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_53 = paddle._C_ops.stack(combine_236, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_53 = paddle._C_ops.full_with_tensor(full_658, stack_53, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_659 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_660 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_661 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_237 = [full_659, slice_207, full_660]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_54 = paddle._C_ops.stack(combine_237, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_54 = paddle._C_ops.full_with_tensor(full_661, stack_54, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_159 = paddle._C_ops.transpose(embedding_26, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_238 = [full_with_tensor_53, full_with_tensor_54]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_239 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_662 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__104, rnn__105, rnn__106, rnn__107 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_159, combine_238, combine_239, None, full_662, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_160 = paddle._C_ops.transpose(rnn__104, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_161 = paddle._C_ops.transpose(transpose_160, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_663 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_664 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_240 = [slice_206, full_663, full_664]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_240]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_665 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_666 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_241 = [slice_206, full_665, full_666]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_241]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_162 = paddle._C_ops.transpose(transpose_161, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_102 = paddle._C_ops.matmul(transpose_162, reshape_120, False, False)

        # pd_op.full: (1xf32) <- ()
        full_667 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__27 = paddle._C_ops.scale_(matmul_102, full_667, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_108 = paddle._C_ops.shape(scale__27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_403 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_404 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_208 = paddle._C_ops.slice(shape_108, [0], full_int_array_403, full_int_array_404, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_668 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_669 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_670 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_242 = [slice_208, full_668, full_669, full_670]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__27, [x.reshape([]) for x in combine_242]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_109 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_405 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_406 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_209 = paddle._C_ops.slice(shape_109, [0], full_int_array_405, full_int_array_406, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_671 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_27 = paddle._C_ops.scale(slice_209, full_671, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_672 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_77 = paddle._C_ops.cast(scale_27, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_673 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_26 = paddle.arange(full_672, cast_77, full_673, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_674 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_78 = paddle._C_ops.cast(slice_209, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_26 = paddle._C_ops.memcpy_h2d(cast_78, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_26 = paddle._C_ops.less_than(full_674, memcpy_h2d_26)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_675 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_78 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_79 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_80 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi64, xi32, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_10878 = 0
        while less_than_26:
            less_than_26, full_674, assign_value_79, assign_value_78, reshape__110, full_675, assign_value_80, = self.pd_op_while_10878_0_0(arange_26, feed_2, slice_209, less_than_26, full_674, assign_value_79, assign_value_78, reshape__110, full_675, assign_value_80)
            while_loop_counter_10878 += 1
            if while_loop_counter_10878 > kWhileLoopLimit:
                break
            
        while_156, while_157, while_158, while_159, while_160, while_161, = full_674, assign_value_79, assign_value_78, reshape__110, full_675, assign_value_80,

        # pd_op.full: (1xi32) <- ()
        full_676 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_677 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_243 = [slice_208, full_676, full_677]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_160, [x.reshape([]) for x in combine_243]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__51 = paddle._C_ops.softmax_(reshape__112, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_163 = paddle._C_ops.transpose(reshape_122, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_103 = paddle._C_ops.matmul(softmax__51, transpose_163, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_164 = paddle._C_ops.transpose(matmul_103, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_407 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_408 = [26]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_210 = paddle._C_ops.slice(transpose_164, [2], full_int_array_407, full_int_array_408, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_409 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_410 = [26]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_211 = paddle._C_ops.slice(transpose_8, [1], full_int_array_409, full_int_array_410, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_244 = [slice_210, slice_211]

        # pd_op.full: (1xi32) <- ()
        full_678 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_244, full_678)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_104 = paddle._C_ops.matmul(concat_25, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add_(matmul_104, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_679 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(add__70, 2, full_679)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_212 = split_with_num_25[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__25 = paddle._C_ops.sigmoid_(slice_212)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_213 = split_with_num_25[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__25 = paddle._C_ops.multiply_(slice_213, sigmoid__25)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_105 = paddle._C_ops.matmul(multiply__25, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__71 = paddle._C_ops.add_(matmul_105, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__52 = paddle._C_ops.softmax_(add__71, -1)

        # pd_op.full: (1xi64) <- ()
        full_680 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_25 = paddle._C_ops.argmax(softmax__52, full_680, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_79 = paddle._C_ops.cast(argmax_25, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_411 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_412 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_413 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__25 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__24, cast_79, full_int_array_411, full_int_array_412, full_int_array_413, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_27 = paddle._C_ops.embedding(set_value_with_tensor__25, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_110 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_414 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_415 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_214 = paddle._C_ops.slice(shape_110, [0], full_int_array_414, full_int_array_415, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_111 = paddle._C_ops.shape(embedding_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_416 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_417 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_215 = paddle._C_ops.slice(shape_111, [0], full_int_array_416, full_int_array_417, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_681 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_682 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_683 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_245 = [full_681, slice_215, full_682]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_55 = paddle._C_ops.stack(combine_245, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_55 = paddle._C_ops.full_with_tensor(full_683, stack_55, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_684 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_685 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_686 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_246 = [full_684, slice_215, full_685]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_56 = paddle._C_ops.stack(combine_246, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_56 = paddle._C_ops.full_with_tensor(full_686, stack_56, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_165 = paddle._C_ops.transpose(embedding_27, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_247 = [full_with_tensor_55, full_with_tensor_56]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_248 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_687 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__108, rnn__109, rnn__110, rnn__111 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_165, combine_247, combine_248, None, full_687, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_166 = paddle._C_ops.transpose(rnn__108, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_167 = paddle._C_ops.transpose(transpose_166, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_688 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_689 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_249 = [slice_214, full_688, full_689]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_249]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_690 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_691 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_250 = [slice_214, full_690, full_691]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_126, reshape_127 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_250]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_168 = paddle._C_ops.transpose(transpose_167, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_106 = paddle._C_ops.matmul(transpose_168, reshape_124, False, False)

        # pd_op.full: (1xf32) <- ()
        full_692 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__28 = paddle._C_ops.scale_(matmul_106, full_692, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_112 = paddle._C_ops.shape(scale__28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_418 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_419 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_216 = paddle._C_ops.slice(shape_112, [0], full_int_array_418, full_int_array_419, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_693 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_694 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_695 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_251 = [slice_216, full_693, full_694, full_695]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__28, [x.reshape([]) for x in combine_251]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_113 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_420 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_421 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_217 = paddle._C_ops.slice(shape_113, [0], full_int_array_420, full_int_array_421, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_696 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_28 = paddle._C_ops.scale(slice_217, full_696, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_697 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_80 = paddle._C_ops.cast(scale_28, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_698 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_27 = paddle.arange(full_697, cast_80, full_698, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_699 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_81 = paddle._C_ops.cast(slice_217, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_27 = paddle._C_ops.memcpy_h2d(cast_81, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_27 = paddle._C_ops.less_than(full_699, memcpy_h2d_27)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_700 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_81 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_82 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_83 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, -1x40x6x40xf32, xi64, xf32, xi32) <- (xb, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi64, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11032 = 0
        while less_than_27:
            less_than_27, full_700, full_699, reshape__114, assign_value_83, assign_value_81, assign_value_82, = self.pd_op_while_11032_0_0(arange_27, feed_2, slice_217, less_than_27, full_700, full_699, reshape__114, assign_value_83, assign_value_81, assign_value_82)
            while_loop_counter_11032 += 1
            if while_loop_counter_11032 > kWhileLoopLimit:
                break
            
        while_162, while_163, while_164, while_165, while_166, while_167, = full_700, full_699, reshape__114, assign_value_83, assign_value_81, assign_value_82,

        # pd_op.full: (1xi32) <- ()
        full_701 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_702 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_252 = [slice_216, full_701, full_702]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_162, [x.reshape([]) for x in combine_252]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__53 = paddle._C_ops.softmax_(reshape__116, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_169 = paddle._C_ops.transpose(reshape_126, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_107 = paddle._C_ops.matmul(softmax__53, transpose_169, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_170 = paddle._C_ops.transpose(matmul_107, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_422 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_423 = [27]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_218 = paddle._C_ops.slice(transpose_170, [2], full_int_array_422, full_int_array_423, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_424 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_425 = [27]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_219 = paddle._C_ops.slice(transpose_8, [1], full_int_array_424, full_int_array_425, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_253 = [slice_218, slice_219]

        # pd_op.full: (1xi32) <- ()
        full_703 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_253, full_703)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_108 = paddle._C_ops.matmul(concat_26, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__72 = paddle._C_ops.add_(matmul_108, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_704 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(add__72, 2, full_704)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_220 = split_with_num_26[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__26 = paddle._C_ops.sigmoid_(slice_220)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_221 = split_with_num_26[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__26 = paddle._C_ops.multiply_(slice_221, sigmoid__26)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_109 = paddle._C_ops.matmul(multiply__26, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__73 = paddle._C_ops.add_(matmul_109, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__54 = paddle._C_ops.softmax_(add__73, -1)

        # pd_op.full: (1xi64) <- ()
        full_705 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_26 = paddle._C_ops.argmax(softmax__54, full_705, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_82 = paddle._C_ops.cast(argmax_26, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_426 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_427 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_428 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__26 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__25, cast_82, full_int_array_426, full_int_array_427, full_int_array_428, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_28 = paddle._C_ops.embedding(set_value_with_tensor__26, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_114 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_429 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_430 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_222 = paddle._C_ops.slice(shape_114, [0], full_int_array_429, full_int_array_430, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_115 = paddle._C_ops.shape(embedding_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_431 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_432 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_223 = paddle._C_ops.slice(shape_115, [0], full_int_array_431, full_int_array_432, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_706 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_707 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_708 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_254 = [full_706, slice_223, full_707]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_57 = paddle._C_ops.stack(combine_254, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_57 = paddle._C_ops.full_with_tensor(full_708, stack_57, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_709 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_710 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_711 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_255 = [full_709, slice_223, full_710]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_58 = paddle._C_ops.stack(combine_255, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_58 = paddle._C_ops.full_with_tensor(full_711, stack_58, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_171 = paddle._C_ops.transpose(embedding_28, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_256 = [full_with_tensor_57, full_with_tensor_58]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_257 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_712 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__112, rnn__113, rnn__114, rnn__115 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_171, combine_256, combine_257, None, full_712, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_172 = paddle._C_ops.transpose(rnn__112, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_173 = paddle._C_ops.transpose(transpose_172, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_713 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_714 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_258 = [slice_222, full_713, full_714]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_128, reshape_129 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_258]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_715 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_716 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_259 = [slice_222, full_715, full_716]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_130, reshape_131 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_259]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_174 = paddle._C_ops.transpose(transpose_173, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_110 = paddle._C_ops.matmul(transpose_174, reshape_128, False, False)

        # pd_op.full: (1xf32) <- ()
        full_717 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__29 = paddle._C_ops.scale_(matmul_110, full_717, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_116 = paddle._C_ops.shape(scale__29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_433 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_434 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_224 = paddle._C_ops.slice(shape_116, [0], full_int_array_433, full_int_array_434, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_718 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_719 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_720 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_260 = [slice_224, full_718, full_719, full_720]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__29, [x.reshape([]) for x in combine_260]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_117 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_435 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_436 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_225 = paddle._C_ops.slice(shape_117, [0], full_int_array_435, full_int_array_436, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_721 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_29 = paddle._C_ops.scale(slice_225, full_721, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_722 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_83 = paddle._C_ops.cast(scale_29, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_723 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_28 = paddle.arange(full_722, cast_83, full_723, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_724 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_84 = paddle._C_ops.cast(slice_225, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_28 = paddle._C_ops.memcpy_h2d(cast_84, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_28 = paddle._C_ops.less_than(full_724, memcpy_h2d_28)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_725 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_84 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_85 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_86 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xf32, xi32, xi64, -1x40x6x40xf32) <- (xb, xi64, -1x40x6x40xf32, xf32, xi32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11186 = 0
        while less_than_28:
            less_than_28, assign_value_86, reshape__118, assign_value_84, assign_value_85, full_724, full_725, = self.pd_op_while_11186_0_0(arange_28, feed_2, slice_225, less_than_28, assign_value_86, reshape__118, assign_value_84, assign_value_85, full_724, full_725)
            while_loop_counter_11186 += 1
            if while_loop_counter_11186 > kWhileLoopLimit:
                break
            
        while_168, while_169, while_170, while_171, while_172, while_173, = assign_value_86, reshape__118, assign_value_84, assign_value_85, full_724, full_725,

        # pd_op.full: (1xi32) <- ()
        full_726 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_727 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_261 = [slice_224, full_726, full_727]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_173, [x.reshape([]) for x in combine_261]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__55 = paddle._C_ops.softmax_(reshape__120, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_175 = paddle._C_ops.transpose(reshape_130, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_111 = paddle._C_ops.matmul(softmax__55, transpose_175, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_176 = paddle._C_ops.transpose(matmul_111, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_437 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_438 = [28]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_226 = paddle._C_ops.slice(transpose_176, [2], full_int_array_437, full_int_array_438, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_439 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_440 = [28]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_227 = paddle._C_ops.slice(transpose_8, [1], full_int_array_439, full_int_array_440, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_262 = [slice_226, slice_227]

        # pd_op.full: (1xi32) <- ()
        full_728 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_262, full_728)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_112 = paddle._C_ops.matmul(concat_27, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__74 = paddle._C_ops.add_(matmul_112, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_729 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(add__74, 2, full_729)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_228 = split_with_num_27[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__27 = paddle._C_ops.sigmoid_(slice_228)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_229 = split_with_num_27[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__27 = paddle._C_ops.multiply_(slice_229, sigmoid__27)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_113 = paddle._C_ops.matmul(multiply__27, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__75 = paddle._C_ops.add_(matmul_113, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__56 = paddle._C_ops.softmax_(add__75, -1)

        # pd_op.full: (1xi64) <- ()
        full_730 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_27 = paddle._C_ops.argmax(softmax__56, full_730, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_85 = paddle._C_ops.cast(argmax_27, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_441 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_442 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_443 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__27 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__26, cast_85, full_int_array_441, full_int_array_442, full_int_array_443, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_29 = paddle._C_ops.embedding(set_value_with_tensor__27, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_118 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_444 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_445 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_230 = paddle._C_ops.slice(shape_118, [0], full_int_array_444, full_int_array_445, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_119 = paddle._C_ops.shape(embedding_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_446 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_447 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_231 = paddle._C_ops.slice(shape_119, [0], full_int_array_446, full_int_array_447, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_731 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_732 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_733 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_263 = [full_731, slice_231, full_732]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_59 = paddle._C_ops.stack(combine_263, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_59 = paddle._C_ops.full_with_tensor(full_733, stack_59, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_734 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_735 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_736 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_264 = [full_734, slice_231, full_735]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_60 = paddle._C_ops.stack(combine_264, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_60 = paddle._C_ops.full_with_tensor(full_736, stack_60, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_177 = paddle._C_ops.transpose(embedding_29, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_265 = [full_with_tensor_59, full_with_tensor_60]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_266 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_737 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__116, rnn__117, rnn__118, rnn__119 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_177, combine_265, combine_266, None, full_737, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_178 = paddle._C_ops.transpose(rnn__116, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_179 = paddle._C_ops.transpose(transpose_178, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_738 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_739 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_267 = [slice_230, full_738, full_739]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_132, reshape_133 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_267]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_740 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_741 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_268 = [slice_230, full_740, full_741]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_134, reshape_135 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_268]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_180 = paddle._C_ops.transpose(transpose_179, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_114 = paddle._C_ops.matmul(transpose_180, reshape_132, False, False)

        # pd_op.full: (1xf32) <- ()
        full_742 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__30 = paddle._C_ops.scale_(matmul_114, full_742, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_120 = paddle._C_ops.shape(scale__30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_448 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_449 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_232 = paddle._C_ops.slice(shape_120, [0], full_int_array_448, full_int_array_449, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_743 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_744 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_745 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_269 = [slice_232, full_743, full_744, full_745]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__30, [x.reshape([]) for x in combine_269]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_121 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_450 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_451 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_233 = paddle._C_ops.slice(shape_121, [0], full_int_array_450, full_int_array_451, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_746 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_30 = paddle._C_ops.scale(slice_233, full_746, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_747 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_86 = paddle._C_ops.cast(scale_30, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_748 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_29 = paddle.arange(full_747, cast_86, full_748, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_749 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_87 = paddle._C_ops.cast(slice_233, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_29 = paddle._C_ops.memcpy_h2d(cast_87, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_29 = paddle._C_ops.less_than(full_749, memcpy_h2d_29)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_750 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_87 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_88 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_89 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xf32, xi64, -1x40x6x40xf32, xi32, -1x40x6x40xf32, xi64) <- (xb, xf32, xi64, -1x40x6x40xf32, xi32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11340 = 0
        while less_than_29:
            less_than_29, assign_value_87, assign_value_89, full_750, assign_value_88, reshape__122, full_749, = self.pd_op_while_11340_0_0(arange_29, feed_2, slice_233, less_than_29, assign_value_87, assign_value_89, full_750, assign_value_88, reshape__122, full_749)
            while_loop_counter_11340 += 1
            if while_loop_counter_11340 > kWhileLoopLimit:
                break
            
        while_174, while_175, while_176, while_177, while_178, while_179, = assign_value_87, assign_value_89, full_750, assign_value_88, reshape__122, full_749,

        # pd_op.full: (1xi32) <- ()
        full_751 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_752 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_270 = [slice_232, full_751, full_752]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_176, [x.reshape([]) for x in combine_270]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__57 = paddle._C_ops.softmax_(reshape__124, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_181 = paddle._C_ops.transpose(reshape_134, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_115 = paddle._C_ops.matmul(softmax__57, transpose_181, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_182 = paddle._C_ops.transpose(matmul_115, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_452 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_453 = [29]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_234 = paddle._C_ops.slice(transpose_182, [2], full_int_array_452, full_int_array_453, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_454 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_455 = [29]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_235 = paddle._C_ops.slice(transpose_8, [1], full_int_array_454, full_int_array_455, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_271 = [slice_234, slice_235]

        # pd_op.full: (1xi32) <- ()
        full_753 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_271, full_753)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_116 = paddle._C_ops.matmul(concat_28, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__76 = paddle._C_ops.add_(matmul_116, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_754 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(add__76, 2, full_754)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_236 = split_with_num_28[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__28 = paddle._C_ops.sigmoid_(slice_236)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_237 = split_with_num_28[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__28 = paddle._C_ops.multiply_(slice_237, sigmoid__28)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_117 = paddle._C_ops.matmul(multiply__28, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__77 = paddle._C_ops.add_(matmul_117, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__58 = paddle._C_ops.softmax_(add__77, -1)

        # pd_op.full: (1xi64) <- ()
        full_755 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_28 = paddle._C_ops.argmax(softmax__58, full_755, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_88 = paddle._C_ops.cast(argmax_28, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_456 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_457 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_458 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__28 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__27, cast_88, full_int_array_456, full_int_array_457, full_int_array_458, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_30 = paddle._C_ops.embedding(set_value_with_tensor__28, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_122 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_459 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_460 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_238 = paddle._C_ops.slice(shape_122, [0], full_int_array_459, full_int_array_460, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_123 = paddle._C_ops.shape(embedding_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_461 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_462 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_239 = paddle._C_ops.slice(shape_123, [0], full_int_array_461, full_int_array_462, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_756 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_757 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_758 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_272 = [full_756, slice_239, full_757]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_61 = paddle._C_ops.stack(combine_272, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_61 = paddle._C_ops.full_with_tensor(full_758, stack_61, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_759 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_760 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_761 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_273 = [full_759, slice_239, full_760]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_62 = paddle._C_ops.stack(combine_273, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_62 = paddle._C_ops.full_with_tensor(full_761, stack_62, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_183 = paddle._C_ops.transpose(embedding_30, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_274 = [full_with_tensor_61, full_with_tensor_62]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_275 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_762 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__120, rnn__121, rnn__122, rnn__123 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_183, combine_274, combine_275, None, full_762, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_184 = paddle._C_ops.transpose(rnn__120, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_185 = paddle._C_ops.transpose(transpose_184, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_763 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_764 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_276 = [slice_238, full_763, full_764]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_136, reshape_137 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_276]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_765 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_766 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_277 = [slice_238, full_765, full_766]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_138, reshape_139 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_277]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_186 = paddle._C_ops.transpose(transpose_185, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_118 = paddle._C_ops.matmul(transpose_186, reshape_136, False, False)

        # pd_op.full: (1xf32) <- ()
        full_767 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__31 = paddle._C_ops.scale_(matmul_118, full_767, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_124 = paddle._C_ops.shape(scale__31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_463 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_464 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_240 = paddle._C_ops.slice(shape_124, [0], full_int_array_463, full_int_array_464, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_768 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_769 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_770 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_278 = [slice_240, full_768, full_769, full_770]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__31, [x.reshape([]) for x in combine_278]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_125 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_465 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_466 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_241 = paddle._C_ops.slice(shape_125, [0], full_int_array_465, full_int_array_466, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_771 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_31 = paddle._C_ops.scale(slice_241, full_771, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_772 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_89 = paddle._C_ops.cast(scale_31, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_773 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_30 = paddle.arange(full_772, cast_89, full_773, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_774 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_90 = paddle._C_ops.cast(slice_241, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_30 = paddle._C_ops.memcpy_h2d(cast_90, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_30 = paddle._C_ops.less_than(full_774, memcpy_h2d_30)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_775 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_90 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_91 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_92 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, xi64, xi64, xf32, -1x40x6x40xf32) <- (xb, xi32, -1x40x6x40xf32, xi64, xi64, xf32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11494 = 0
        while less_than_30:
            less_than_30, assign_value_91, full_775, assign_value_92, full_774, assign_value_90, reshape__126, = self.pd_op_while_11494_0_0(arange_30, feed_2, slice_241, less_than_30, assign_value_91, full_775, assign_value_92, full_774, assign_value_90, reshape__126)
            while_loop_counter_11494 += 1
            if while_loop_counter_11494 > kWhileLoopLimit:
                break
            
        while_180, while_181, while_182, while_183, while_184, while_185, = assign_value_91, full_775, assign_value_92, full_774, assign_value_90, reshape__126,

        # pd_op.full: (1xi32) <- ()
        full_776 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_777 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_279 = [slice_240, full_776, full_777]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_181, [x.reshape([]) for x in combine_279]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__59 = paddle._C_ops.softmax_(reshape__128, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_187 = paddle._C_ops.transpose(reshape_138, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_119 = paddle._C_ops.matmul(softmax__59, transpose_187, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_188 = paddle._C_ops.transpose(matmul_119, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_467 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_468 = [30]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_242 = paddle._C_ops.slice(transpose_188, [2], full_int_array_467, full_int_array_468, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_469 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_470 = [30]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_243 = paddle._C_ops.slice(transpose_8, [1], full_int_array_469, full_int_array_470, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_280 = [slice_242, slice_243]

        # pd_op.full: (1xi32) <- ()
        full_778 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_280, full_778)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_120 = paddle._C_ops.matmul(concat_29, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__78 = paddle._C_ops.add_(matmul_120, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_779 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(add__78, 2, full_779)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_244 = split_with_num_29[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__29 = paddle._C_ops.sigmoid_(slice_244)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_245 = split_with_num_29[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__29 = paddle._C_ops.multiply_(slice_245, sigmoid__29)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_121 = paddle._C_ops.matmul(multiply__29, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__79 = paddle._C_ops.add_(matmul_121, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__60 = paddle._C_ops.softmax_(add__79, -1)

        # pd_op.full: (1xi64) <- ()
        full_780 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_29 = paddle._C_ops.argmax(softmax__60, full_780, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_91 = paddle._C_ops.cast(argmax_29, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_471 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_472 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_473 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__29 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__28, cast_91, full_int_array_471, full_int_array_472, full_int_array_473, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_31 = paddle._C_ops.embedding(set_value_with_tensor__29, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_126 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_474 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_475 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_246 = paddle._C_ops.slice(shape_126, [0], full_int_array_474, full_int_array_475, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_127 = paddle._C_ops.shape(embedding_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_476 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_477 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_247 = paddle._C_ops.slice(shape_127, [0], full_int_array_476, full_int_array_477, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_781 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_782 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_783 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_281 = [full_781, slice_247, full_782]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_63 = paddle._C_ops.stack(combine_281, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_63 = paddle._C_ops.full_with_tensor(full_783, stack_63, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_784 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_785 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_786 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_282 = [full_784, slice_247, full_785]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_64 = paddle._C_ops.stack(combine_282, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_64 = paddle._C_ops.full_with_tensor(full_786, stack_64, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_189 = paddle._C_ops.transpose(embedding_31, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_283 = [full_with_tensor_63, full_with_tensor_64]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_284 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_787 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__124, rnn__125, rnn__126, rnn__127 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_189, combine_283, combine_284, None, full_787, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_190 = paddle._C_ops.transpose(rnn__124, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_191 = paddle._C_ops.transpose(transpose_190, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_788 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_789 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_285 = [slice_246, full_788, full_789]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_140, reshape_141 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_285]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_790 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_791 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_286 = [slice_246, full_790, full_791]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_142, reshape_143 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_286]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_192 = paddle._C_ops.transpose(transpose_191, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_122 = paddle._C_ops.matmul(transpose_192, reshape_140, False, False)

        # pd_op.full: (1xf32) <- ()
        full_792 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__32 = paddle._C_ops.scale_(matmul_122, full_792, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_128 = paddle._C_ops.shape(scale__32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_478 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_479 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_248 = paddle._C_ops.slice(shape_128, [0], full_int_array_478, full_int_array_479, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_793 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_794 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_795 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_287 = [slice_248, full_793, full_794, full_795]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__32, [x.reshape([]) for x in combine_287]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_129 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_480 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_481 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_249 = paddle._C_ops.slice(shape_129, [0], full_int_array_480, full_int_array_481, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_796 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_32 = paddle._C_ops.scale(slice_249, full_796, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_797 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_92 = paddle._C_ops.cast(scale_32, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_798 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_31 = paddle.arange(full_797, cast_92, full_798, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_799 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_93 = paddle._C_ops.cast(slice_249, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_31 = paddle._C_ops.memcpy_h2d(cast_93, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_31 = paddle._C_ops.less_than(full_799, memcpy_h2d_31)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_800 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_93 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_94 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_95 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi64) <- (xb, xi32, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11648 = 0
        while less_than_31:
            less_than_31, assign_value_94, full_800, assign_value_95, reshape__130, assign_value_93, full_799, = self.pd_op_while_11648_0_0(arange_31, feed_2, slice_249, less_than_31, assign_value_94, full_800, assign_value_95, reshape__130, assign_value_93, full_799)
            while_loop_counter_11648 += 1
            if while_loop_counter_11648 > kWhileLoopLimit:
                break
            
        while_186, while_187, while_188, while_189, while_190, while_191, = assign_value_94, full_800, assign_value_95, reshape__130, assign_value_93, full_799,

        # pd_op.full: (1xi32) <- ()
        full_801 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_802 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_288 = [slice_248, full_801, full_802]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_187, [x.reshape([]) for x in combine_288]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__61 = paddle._C_ops.softmax_(reshape__132, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_193 = paddle._C_ops.transpose(reshape_142, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_123 = paddle._C_ops.matmul(softmax__61, transpose_193, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_194 = paddle._C_ops.transpose(matmul_123, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_482 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_483 = [31]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_250 = paddle._C_ops.slice(transpose_194, [2], full_int_array_482, full_int_array_483, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_484 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_485 = [31]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_251 = paddle._C_ops.slice(transpose_8, [1], full_int_array_484, full_int_array_485, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_289 = [slice_250, slice_251]

        # pd_op.full: (1xi32) <- ()
        full_803 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_289, full_803)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_124 = paddle._C_ops.matmul(concat_30, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__80 = paddle._C_ops.add_(matmul_124, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_804 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(add__80, 2, full_804)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_252 = split_with_num_30[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__30 = paddle._C_ops.sigmoid_(slice_252)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_253 = split_with_num_30[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__30 = paddle._C_ops.multiply_(slice_253, sigmoid__30)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_125 = paddle._C_ops.matmul(multiply__30, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__81 = paddle._C_ops.add_(matmul_125, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__62 = paddle._C_ops.softmax_(add__81, -1)

        # pd_op.full: (1xi64) <- ()
        full_805 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_30 = paddle._C_ops.argmax(softmax__62, full_805, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_94 = paddle._C_ops.cast(argmax_30, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_486 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_487 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_488 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__30 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__29, cast_94, full_int_array_486, full_int_array_487, full_int_array_488, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_32 = paddle._C_ops.embedding(set_value_with_tensor__30, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_130 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_489 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_490 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_254 = paddle._C_ops.slice(shape_130, [0], full_int_array_489, full_int_array_490, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_131 = paddle._C_ops.shape(embedding_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_491 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_492 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_255 = paddle._C_ops.slice(shape_131, [0], full_int_array_491, full_int_array_492, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_806 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_807 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_808 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_290 = [full_806, slice_255, full_807]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_65 = paddle._C_ops.stack(combine_290, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_65 = paddle._C_ops.full_with_tensor(full_808, stack_65, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_809 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_810 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_811 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_291 = [full_809, slice_255, full_810]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_66 = paddle._C_ops.stack(combine_291, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_66 = paddle._C_ops.full_with_tensor(full_811, stack_66, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_195 = paddle._C_ops.transpose(embedding_32, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_292 = [full_with_tensor_65, full_with_tensor_66]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_293 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_812 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__128, rnn__129, rnn__130, rnn__131 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_195, combine_292, combine_293, None, full_812, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_196 = paddle._C_ops.transpose(rnn__128, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_197 = paddle._C_ops.transpose(transpose_196, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_813 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_814 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_294 = [slice_254, full_813, full_814]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_144, reshape_145 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_294]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_815 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_816 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_295 = [slice_254, full_815, full_816]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_146, reshape_147 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_295]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_198 = paddle._C_ops.transpose(transpose_197, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_126 = paddle._C_ops.matmul(transpose_198, reshape_144, False, False)

        # pd_op.full: (1xf32) <- ()
        full_817 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__33 = paddle._C_ops.scale_(matmul_126, full_817, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_132 = paddle._C_ops.shape(scale__33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_493 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_494 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_256 = paddle._C_ops.slice(shape_132, [0], full_int_array_493, full_int_array_494, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_818 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_819 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_820 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_296 = [slice_256, full_818, full_819, full_820]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__33, [x.reshape([]) for x in combine_296]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_133 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_495 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_496 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_257 = paddle._C_ops.slice(shape_133, [0], full_int_array_495, full_int_array_496, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_821 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_33 = paddle._C_ops.scale(slice_257, full_821, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_822 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_95 = paddle._C_ops.cast(scale_33, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_823 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_32 = paddle.arange(full_822, cast_95, full_823, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_824 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_96 = paddle._C_ops.cast(slice_257, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_32 = paddle._C_ops.memcpy_h2d(cast_96, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_32 = paddle._C_ops.less_than(full_824, memcpy_h2d_32)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_825 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_96 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_97 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_98 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xi64, xf32, xi64, -1x40x6x40xf32, xi32) <- (xb, -1x40x6x40xf32, xi64, xf32, xi64, -1x40x6x40xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11802 = 0
        while less_than_32:
            less_than_32, reshape__134, full_824, assign_value_96, assign_value_98, full_825, assign_value_97, = self.pd_op_while_11802_0_0(arange_32, feed_2, slice_257, less_than_32, reshape__134, full_824, assign_value_96, assign_value_98, full_825, assign_value_97)
            while_loop_counter_11802 += 1
            if while_loop_counter_11802 > kWhileLoopLimit:
                break
            
        while_192, while_193, while_194, while_195, while_196, while_197, = reshape__134, full_824, assign_value_96, assign_value_98, full_825, assign_value_97,

        # pd_op.full: (1xi32) <- ()
        full_826 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_827 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_297 = [slice_256, full_826, full_827]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_196, [x.reshape([]) for x in combine_297]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__63 = paddle._C_ops.softmax_(reshape__136, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_199 = paddle._C_ops.transpose(reshape_146, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_127 = paddle._C_ops.matmul(softmax__63, transpose_199, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_200 = paddle._C_ops.transpose(matmul_127, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_497 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_498 = [32]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_258 = paddle._C_ops.slice(transpose_200, [2], full_int_array_497, full_int_array_498, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_499 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_500 = [32]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_259 = paddle._C_ops.slice(transpose_8, [1], full_int_array_499, full_int_array_500, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_298 = [slice_258, slice_259]

        # pd_op.full: (1xi32) <- ()
        full_828 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_298, full_828)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_128 = paddle._C_ops.matmul(concat_31, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__82 = paddle._C_ops.add_(matmul_128, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_829 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(add__82, 2, full_829)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_260 = split_with_num_31[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__31 = paddle._C_ops.sigmoid_(slice_260)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_261 = split_with_num_31[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__31 = paddle._C_ops.multiply_(slice_261, sigmoid__31)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_129 = paddle._C_ops.matmul(multiply__31, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__83 = paddle._C_ops.add_(matmul_129, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__64 = paddle._C_ops.softmax_(add__83, -1)

        # pd_op.full: (1xi64) <- ()
        full_830 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_31 = paddle._C_ops.argmax(softmax__64, full_830, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_97 = paddle._C_ops.cast(argmax_31, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_501 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_502 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_503 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__31 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__30, cast_97, full_int_array_501, full_int_array_502, full_int_array_503, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_33 = paddle._C_ops.embedding(set_value_with_tensor__31, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_134 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_504 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_505 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_262 = paddle._C_ops.slice(shape_134, [0], full_int_array_504, full_int_array_505, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_135 = paddle._C_ops.shape(embedding_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_506 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_507 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_263 = paddle._C_ops.slice(shape_135, [0], full_int_array_506, full_int_array_507, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_831 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_832 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_833 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_299 = [full_831, slice_263, full_832]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_67 = paddle._C_ops.stack(combine_299, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_67 = paddle._C_ops.full_with_tensor(full_833, stack_67, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_834 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_835 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_836 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_300 = [full_834, slice_263, full_835]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_68 = paddle._C_ops.stack(combine_300, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_68 = paddle._C_ops.full_with_tensor(full_836, stack_68, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_201 = paddle._C_ops.transpose(embedding_33, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_301 = [full_with_tensor_67, full_with_tensor_68]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_302 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_837 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__132, rnn__133, rnn__134, rnn__135 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_201, combine_301, combine_302, None, full_837, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_202 = paddle._C_ops.transpose(rnn__132, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_203 = paddle._C_ops.transpose(transpose_202, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_838 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_839 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_303 = [slice_262, full_838, full_839]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_148, reshape_149 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_303]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_840 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_841 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_304 = [slice_262, full_840, full_841]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_150, reshape_151 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_304]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_204 = paddle._C_ops.transpose(transpose_203, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_130 = paddle._C_ops.matmul(transpose_204, reshape_148, False, False)

        # pd_op.full: (1xf32) <- ()
        full_842 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__34 = paddle._C_ops.scale_(matmul_130, full_842, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_136 = paddle._C_ops.shape(scale__34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_508 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_509 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_264 = paddle._C_ops.slice(shape_136, [0], full_int_array_508, full_int_array_509, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_843 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_844 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_845 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_305 = [slice_264, full_843, full_844, full_845]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__34, [x.reshape([]) for x in combine_305]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_137 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_510 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_511 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_265 = paddle._C_ops.slice(shape_137, [0], full_int_array_510, full_int_array_511, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_846 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_34 = paddle._C_ops.scale(slice_265, full_846, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_847 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_98 = paddle._C_ops.cast(scale_34, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_848 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_33 = paddle.arange(full_847, cast_98, full_848, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_849 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_99 = paddle._C_ops.cast(slice_265, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_33 = paddle._C_ops.memcpy_h2d(cast_99, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_33 = paddle._C_ops.less_than(full_849, memcpy_h2d_33)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_850 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_99 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_100 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_101 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi32) <- (xb, xi64, xi64, -1x40x6x40xf32, -1x40x6x40xf32, xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_11956 = 0
        while less_than_33:
            less_than_33, assign_value_101, full_849, reshape__138, full_850, assign_value_99, assign_value_100, = self.pd_op_while_11956_0_0(arange_33, feed_2, slice_265, less_than_33, assign_value_101, full_849, reshape__138, full_850, assign_value_99, assign_value_100)
            while_loop_counter_11956 += 1
            if while_loop_counter_11956 > kWhileLoopLimit:
                break
            
        while_198, while_199, while_200, while_201, while_202, while_203, = assign_value_101, full_849, reshape__138, full_850, assign_value_99, assign_value_100,

        # pd_op.full: (1xi32) <- ()
        full_851 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_852 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_306 = [slice_264, full_851, full_852]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_201, [x.reshape([]) for x in combine_306]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__65 = paddle._C_ops.softmax_(reshape__140, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_205 = paddle._C_ops.transpose(reshape_150, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_131 = paddle._C_ops.matmul(softmax__65, transpose_205, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_206 = paddle._C_ops.transpose(matmul_131, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_512 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_513 = [33]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_266 = paddle._C_ops.slice(transpose_206, [2], full_int_array_512, full_int_array_513, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_514 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_515 = [33]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_267 = paddle._C_ops.slice(transpose_8, [1], full_int_array_514, full_int_array_515, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_307 = [slice_266, slice_267]

        # pd_op.full: (1xi32) <- ()
        full_853 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_307, full_853)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_132 = paddle._C_ops.matmul(concat_32, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__84 = paddle._C_ops.add_(matmul_132, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_854 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_32 = paddle._C_ops.split_with_num(add__84, 2, full_854)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_268 = split_with_num_32[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__32 = paddle._C_ops.sigmoid_(slice_268)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_269 = split_with_num_32[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__32 = paddle._C_ops.multiply_(slice_269, sigmoid__32)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_133 = paddle._C_ops.matmul(multiply__32, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__85 = paddle._C_ops.add_(matmul_133, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__66 = paddle._C_ops.softmax_(add__85, -1)

        # pd_op.full: (1xi64) <- ()
        full_855 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_32 = paddle._C_ops.argmax(softmax__66, full_855, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_100 = paddle._C_ops.cast(argmax_32, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_516 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_517 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_518 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__32 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__31, cast_100, full_int_array_516, full_int_array_517, full_int_array_518, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_34 = paddle._C_ops.embedding(set_value_with_tensor__32, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_138 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_519 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_520 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_270 = paddle._C_ops.slice(shape_138, [0], full_int_array_519, full_int_array_520, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_139 = paddle._C_ops.shape(embedding_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_521 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_522 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_271 = paddle._C_ops.slice(shape_139, [0], full_int_array_521, full_int_array_522, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_856 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_857 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_858 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_308 = [full_856, slice_271, full_857]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_69 = paddle._C_ops.stack(combine_308, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_69 = paddle._C_ops.full_with_tensor(full_858, stack_69, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_859 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_860 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_861 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_309 = [full_859, slice_271, full_860]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_70 = paddle._C_ops.stack(combine_309, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_70 = paddle._C_ops.full_with_tensor(full_861, stack_70, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_207 = paddle._C_ops.transpose(embedding_34, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_310 = [full_with_tensor_69, full_with_tensor_70]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_311 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_862 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__136, rnn__137, rnn__138, rnn__139 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_207, combine_310, combine_311, None, full_862, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_208 = paddle._C_ops.transpose(rnn__136, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_209 = paddle._C_ops.transpose(transpose_208, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_863 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_864 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_312 = [slice_270, full_863, full_864]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_152, reshape_153 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_312]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_865 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_866 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_313 = [slice_270, full_865, full_866]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_154, reshape_155 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_313]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_210 = paddle._C_ops.transpose(transpose_209, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_134 = paddle._C_ops.matmul(transpose_210, reshape_152, False, False)

        # pd_op.full: (1xf32) <- ()
        full_867 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__35 = paddle._C_ops.scale_(matmul_134, full_867, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_140 = paddle._C_ops.shape(scale__35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_523 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_524 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_272 = paddle._C_ops.slice(shape_140, [0], full_int_array_523, full_int_array_524, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_868 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_869 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_870 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_314 = [slice_272, full_868, full_869, full_870]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__35, [x.reshape([]) for x in combine_314]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_141 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_525 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_526 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_273 = paddle._C_ops.slice(shape_141, [0], full_int_array_525, full_int_array_526, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_871 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_35 = paddle._C_ops.scale(slice_273, full_871, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_872 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_101 = paddle._C_ops.cast(scale_35, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_873 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_34 = paddle.arange(full_872, cast_101, full_873, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_874 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_102 = paddle._C_ops.cast(slice_273, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_34 = paddle._C_ops.memcpy_h2d(cast_102, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_34 = paddle._C_ops.less_than(full_874, memcpy_h2d_34)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_875 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_102 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_103 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_104 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi32) <- (xb, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12110 = 0
        while less_than_34:
            less_than_34, assign_value_104, assign_value_102, reshape__142, full_875, full_874, assign_value_103, = self.pd_op_while_12110_0_0(arange_34, feed_2, slice_273, less_than_34, assign_value_104, assign_value_102, reshape__142, full_875, full_874, assign_value_103)
            while_loop_counter_12110 += 1
            if while_loop_counter_12110 > kWhileLoopLimit:
                break
            
        while_204, while_205, while_206, while_207, while_208, while_209, = assign_value_104, assign_value_102, reshape__142, full_875, full_874, assign_value_103,

        # pd_op.full: (1xi32) <- ()
        full_876 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_877 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_315 = [slice_272, full_876, full_877]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_207, [x.reshape([]) for x in combine_315]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__67 = paddle._C_ops.softmax_(reshape__144, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_211 = paddle._C_ops.transpose(reshape_154, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_135 = paddle._C_ops.matmul(softmax__67, transpose_211, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_212 = paddle._C_ops.transpose(matmul_135, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_527 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_528 = [34]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_274 = paddle._C_ops.slice(transpose_212, [2], full_int_array_527, full_int_array_528, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_529 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_530 = [34]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_275 = paddle._C_ops.slice(transpose_8, [1], full_int_array_529, full_int_array_530, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_316 = [slice_274, slice_275]

        # pd_op.full: (1xi32) <- ()
        full_878 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_316, full_878)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_136 = paddle._C_ops.matmul(concat_33, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__86 = paddle._C_ops.add_(matmul_136, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_879 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_33 = paddle._C_ops.split_with_num(add__86, 2, full_879)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_276 = split_with_num_33[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__33 = paddle._C_ops.sigmoid_(slice_276)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_277 = split_with_num_33[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__33 = paddle._C_ops.multiply_(slice_277, sigmoid__33)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_137 = paddle._C_ops.matmul(multiply__33, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__87 = paddle._C_ops.add_(matmul_137, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__68 = paddle._C_ops.softmax_(add__87, -1)

        # pd_op.full: (1xi64) <- ()
        full_880 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_33 = paddle._C_ops.argmax(softmax__68, full_880, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_103 = paddle._C_ops.cast(argmax_33, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_531 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_532 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_533 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__33 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__32, cast_103, full_int_array_531, full_int_array_532, full_int_array_533, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_35 = paddle._C_ops.embedding(set_value_with_tensor__33, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_142 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_534 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_535 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_278 = paddle._C_ops.slice(shape_142, [0], full_int_array_534, full_int_array_535, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_143 = paddle._C_ops.shape(embedding_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_536 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_537 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_279 = paddle._C_ops.slice(shape_143, [0], full_int_array_536, full_int_array_537, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_881 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_882 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_883 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_317 = [full_881, slice_279, full_882]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_71 = paddle._C_ops.stack(combine_317, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_71 = paddle._C_ops.full_with_tensor(full_883, stack_71, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_884 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_885 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_886 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_318 = [full_884, slice_279, full_885]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_72 = paddle._C_ops.stack(combine_318, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_72 = paddle._C_ops.full_with_tensor(full_886, stack_72, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_213 = paddle._C_ops.transpose(embedding_35, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_319 = [full_with_tensor_71, full_with_tensor_72]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_320 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_887 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__140, rnn__141, rnn__142, rnn__143 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_213, combine_319, combine_320, None, full_887, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_214 = paddle._C_ops.transpose(rnn__140, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_215 = paddle._C_ops.transpose(transpose_214, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_888 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_889 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_321 = [slice_278, full_888, full_889]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_156, reshape_157 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_321]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_890 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_891 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_322 = [slice_278, full_890, full_891]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_158, reshape_159 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_322]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_216 = paddle._C_ops.transpose(transpose_215, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_138 = paddle._C_ops.matmul(transpose_216, reshape_156, False, False)

        # pd_op.full: (1xf32) <- ()
        full_892 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__36 = paddle._C_ops.scale_(matmul_138, full_892, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_144 = paddle._C_ops.shape(scale__36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_538 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_539 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_280 = paddle._C_ops.slice(shape_144, [0], full_int_array_538, full_int_array_539, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_893 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_894 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_895 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_323 = [slice_280, full_893, full_894, full_895]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__36, [x.reshape([]) for x in combine_323]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_145 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_540 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_541 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_281 = paddle._C_ops.slice(shape_145, [0], full_int_array_540, full_int_array_541, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_896 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_36 = paddle._C_ops.scale(slice_281, full_896, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_897 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_104 = paddle._C_ops.cast(scale_36, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_898 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_35 = paddle.arange(full_897, cast_104, full_898, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_899 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_105 = paddle._C_ops.cast(slice_281, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_35 = paddle._C_ops.memcpy_h2d(cast_105, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_35 = paddle._C_ops.less_than(full_899, memcpy_h2d_35)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_900 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_105 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_106 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_107 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64) <- (xb, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32, xi32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12264 = 0
        while less_than_35:
            less_than_35, full_900, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_899, = self.pd_op_while_12264_0_0(arange_35, feed_2, slice_281, less_than_35, full_900, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_899)
            while_loop_counter_12264 += 1
            if while_loop_counter_12264 > kWhileLoopLimit:
                break
            
        while_210, while_211, while_212, while_213, while_214, while_215, = full_900, assign_value_105, assign_value_107, reshape__146, assign_value_106, full_899,

        # pd_op.full: (1xi32) <- ()
        full_901 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_902 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_324 = [slice_280, full_901, full_902]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_210, [x.reshape([]) for x in combine_324]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__69 = paddle._C_ops.softmax_(reshape__148, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_217 = paddle._C_ops.transpose(reshape_158, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_139 = paddle._C_ops.matmul(softmax__69, transpose_217, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_218 = paddle._C_ops.transpose(matmul_139, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_542 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_543 = [35]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_282 = paddle._C_ops.slice(transpose_218, [2], full_int_array_542, full_int_array_543, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_544 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_545 = [35]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_283 = paddle._C_ops.slice(transpose_8, [1], full_int_array_544, full_int_array_545, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_325 = [slice_282, slice_283]

        # pd_op.full: (1xi32) <- ()
        full_903 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_325, full_903)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_140 = paddle._C_ops.matmul(concat_34, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__88 = paddle._C_ops.add_(matmul_140, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_904 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_34 = paddle._C_ops.split_with_num(add__88, 2, full_904)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_284 = split_with_num_34[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__34 = paddle._C_ops.sigmoid_(slice_284)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_285 = split_with_num_34[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__34 = paddle._C_ops.multiply_(slice_285, sigmoid__34)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_141 = paddle._C_ops.matmul(multiply__34, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__89 = paddle._C_ops.add_(matmul_141, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__70 = paddle._C_ops.softmax_(add__89, -1)

        # pd_op.full: (1xi64) <- ()
        full_905 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_34 = paddle._C_ops.argmax(softmax__70, full_905, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_106 = paddle._C_ops.cast(argmax_34, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_546 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_547 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_548 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__34 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__33, cast_106, full_int_array_546, full_int_array_547, full_int_array_548, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_36 = paddle._C_ops.embedding(set_value_with_tensor__34, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_146 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_549 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_550 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_286 = paddle._C_ops.slice(shape_146, [0], full_int_array_549, full_int_array_550, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_147 = paddle._C_ops.shape(embedding_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_551 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_552 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_287 = paddle._C_ops.slice(shape_147, [0], full_int_array_551, full_int_array_552, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_906 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_907 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_908 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_326 = [full_906, slice_287, full_907]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_73 = paddle._C_ops.stack(combine_326, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_73 = paddle._C_ops.full_with_tensor(full_908, stack_73, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_909 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_910 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_911 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_327 = [full_909, slice_287, full_910]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_74 = paddle._C_ops.stack(combine_327, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_74 = paddle._C_ops.full_with_tensor(full_911, stack_74, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_219 = paddle._C_ops.transpose(embedding_36, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_328 = [full_with_tensor_73, full_with_tensor_74]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_329 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_912 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__144, rnn__145, rnn__146, rnn__147 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_219, combine_328, combine_329, None, full_912, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_220 = paddle._C_ops.transpose(rnn__144, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_221 = paddle._C_ops.transpose(transpose_220, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_913 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_914 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_330 = [slice_286, full_913, full_914]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_160, reshape_161 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_330]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_915 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_916 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_331 = [slice_286, full_915, full_916]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_162, reshape_163 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_331]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_222 = paddle._C_ops.transpose(transpose_221, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_142 = paddle._C_ops.matmul(transpose_222, reshape_160, False, False)

        # pd_op.full: (1xf32) <- ()
        full_917 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__37 = paddle._C_ops.scale_(matmul_142, full_917, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_148 = paddle._C_ops.shape(scale__37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_553 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_554 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_288 = paddle._C_ops.slice(shape_148, [0], full_int_array_553, full_int_array_554, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_918 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_919 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_920 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_332 = [slice_288, full_918, full_919, full_920]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__37, [x.reshape([]) for x in combine_332]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_149 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_555 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_556 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_289 = paddle._C_ops.slice(shape_149, [0], full_int_array_555, full_int_array_556, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_921 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_37 = paddle._C_ops.scale(slice_289, full_921, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_922 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_107 = paddle._C_ops.cast(scale_37, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_923 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_36 = paddle.arange(full_922, cast_107, full_923, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_924 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_108 = paddle._C_ops.cast(slice_289, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_36 = paddle._C_ops.memcpy_h2d(cast_108, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_36 = paddle._C_ops.less_than(full_924, memcpy_h2d_36)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_925 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_108 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_109 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_110 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1x40x6x40xf32, xf32, xi64, xi64, -1x40x6x40xf32, xi32) <- (xb, -1x40x6x40xf32, xf32, xi64, xi64, -1x40x6x40xf32, xi32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12418 = 0
        while less_than_36:
            less_than_36, reshape__150, assign_value_108, full_924, assign_value_110, full_925, assign_value_109, = self.pd_op_while_12418_0_0(arange_36, feed_2, slice_289, less_than_36, reshape__150, assign_value_108, full_924, assign_value_110, full_925, assign_value_109)
            while_loop_counter_12418 += 1
            if while_loop_counter_12418 > kWhileLoopLimit:
                break
            
        while_216, while_217, while_218, while_219, while_220, while_221, = reshape__150, assign_value_108, full_924, assign_value_110, full_925, assign_value_109,

        # pd_op.full: (1xi32) <- ()
        full_926 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_927 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_333 = [slice_288, full_926, full_927]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_220, [x.reshape([]) for x in combine_333]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__71 = paddle._C_ops.softmax_(reshape__152, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_223 = paddle._C_ops.transpose(reshape_162, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_143 = paddle._C_ops.matmul(softmax__71, transpose_223, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_224 = paddle._C_ops.transpose(matmul_143, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_557 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_558 = [36]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_290 = paddle._C_ops.slice(transpose_224, [2], full_int_array_557, full_int_array_558, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_559 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_560 = [36]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_291 = paddle._C_ops.slice(transpose_8, [1], full_int_array_559, full_int_array_560, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_334 = [slice_290, slice_291]

        # pd_op.full: (1xi32) <- ()
        full_928 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_334, full_928)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_144 = paddle._C_ops.matmul(concat_35, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__90 = paddle._C_ops.add_(matmul_144, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_929 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_35 = paddle._C_ops.split_with_num(add__90, 2, full_929)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_292 = split_with_num_35[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__35 = paddle._C_ops.sigmoid_(slice_292)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_293 = split_with_num_35[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__35 = paddle._C_ops.multiply_(slice_293, sigmoid__35)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_145 = paddle._C_ops.matmul(multiply__35, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__91 = paddle._C_ops.add_(matmul_145, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__72 = paddle._C_ops.softmax_(add__91, -1)

        # pd_op.full: (1xi64) <- ()
        full_930 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_35 = paddle._C_ops.argmax(softmax__72, full_930, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_109 = paddle._C_ops.cast(argmax_35, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_561 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_562 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_563 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__35 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__34, cast_109, full_int_array_561, full_int_array_562, full_int_array_563, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_37 = paddle._C_ops.embedding(set_value_with_tensor__35, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_150 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_564 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_565 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_294 = paddle._C_ops.slice(shape_150, [0], full_int_array_564, full_int_array_565, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_151 = paddle._C_ops.shape(embedding_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_566 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_567 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_295 = paddle._C_ops.slice(shape_151, [0], full_int_array_566, full_int_array_567, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_931 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_932 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_933 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_335 = [full_931, slice_295, full_932]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_75 = paddle._C_ops.stack(combine_335, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_75 = paddle._C_ops.full_with_tensor(full_933, stack_75, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_934 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_935 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_936 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_336 = [full_934, slice_295, full_935]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_76 = paddle._C_ops.stack(combine_336, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_76 = paddle._C_ops.full_with_tensor(full_936, stack_76, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_225 = paddle._C_ops.transpose(embedding_37, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_337 = [full_with_tensor_75, full_with_tensor_76]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_338 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_937 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__148, rnn__149, rnn__150, rnn__151 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_225, combine_337, combine_338, None, full_937, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_226 = paddle._C_ops.transpose(rnn__148, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_227 = paddle._C_ops.transpose(transpose_226, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_938 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_939 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_339 = [slice_294, full_938, full_939]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_164, reshape_165 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_339]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_940 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_941 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_340 = [slice_294, full_940, full_941]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_166, reshape_167 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_340]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_228 = paddle._C_ops.transpose(transpose_227, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_146 = paddle._C_ops.matmul(transpose_228, reshape_164, False, False)

        # pd_op.full: (1xf32) <- ()
        full_942 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__38 = paddle._C_ops.scale_(matmul_146, full_942, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_152 = paddle._C_ops.shape(scale__38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_568 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_569 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_296 = paddle._C_ops.slice(shape_152, [0], full_int_array_568, full_int_array_569, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_943 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_944 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_945 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_341 = [slice_296, full_943, full_944, full_945]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__38, [x.reshape([]) for x in combine_341]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_153 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_570 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_571 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_297 = paddle._C_ops.slice(shape_153, [0], full_int_array_570, full_int_array_571, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_946 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_38 = paddle._C_ops.scale(slice_297, full_946, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_947 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_110 = paddle._C_ops.cast(scale_38, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_948 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_37 = paddle.arange(full_947, cast_110, full_948, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_949 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_111 = paddle._C_ops.cast(slice_297, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_37 = paddle._C_ops.memcpy_h2d(cast_111, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_37 = paddle._C_ops.less_than(full_949, memcpy_h2d_37)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_950 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_111 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_112 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_113 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32) <- (xb, xi32, xi64, -1x40x6x40xf32, xf32, xi64, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12572 = 0
        while less_than_37:
            less_than_37, assign_value_112, full_949, full_950, assign_value_111, assign_value_113, reshape__154, = self.pd_op_while_12572_0_0(arange_37, feed_2, slice_297, less_than_37, assign_value_112, full_949, full_950, assign_value_111, assign_value_113, reshape__154)
            while_loop_counter_12572 += 1
            if while_loop_counter_12572 > kWhileLoopLimit:
                break
            
        while_222, while_223, while_224, while_225, while_226, while_227, = assign_value_112, full_949, full_950, assign_value_111, assign_value_113, reshape__154,

        # pd_op.full: (1xi32) <- ()
        full_951 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_952 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_342 = [slice_296, full_951, full_952]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_224, [x.reshape([]) for x in combine_342]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__73 = paddle._C_ops.softmax_(reshape__156, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_229 = paddle._C_ops.transpose(reshape_166, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_147 = paddle._C_ops.matmul(softmax__73, transpose_229, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_230 = paddle._C_ops.transpose(matmul_147, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_572 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_573 = [37]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_298 = paddle._C_ops.slice(transpose_230, [2], full_int_array_572, full_int_array_573, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_574 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_575 = [37]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_299 = paddle._C_ops.slice(transpose_8, [1], full_int_array_574, full_int_array_575, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_343 = [slice_298, slice_299]

        # pd_op.full: (1xi32) <- ()
        full_953 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_343, full_953)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_148 = paddle._C_ops.matmul(concat_36, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__92 = paddle._C_ops.add_(matmul_148, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_954 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_36 = paddle._C_ops.split_with_num(add__92, 2, full_954)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_300 = split_with_num_36[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__36 = paddle._C_ops.sigmoid_(slice_300)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_301 = split_with_num_36[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__36 = paddle._C_ops.multiply_(slice_301, sigmoid__36)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_149 = paddle._C_ops.matmul(multiply__36, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__93 = paddle._C_ops.add_(matmul_149, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__74 = paddle._C_ops.softmax_(add__93, -1)

        # pd_op.full: (1xi64) <- ()
        full_955 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_36 = paddle._C_ops.argmax(softmax__74, full_955, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_112 = paddle._C_ops.cast(argmax_36, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_576 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_577 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_578 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__36 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__35, cast_112, full_int_array_576, full_int_array_577, full_int_array_578, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_38 = paddle._C_ops.embedding(set_value_with_tensor__36, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_154 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_579 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_580 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_302 = paddle._C_ops.slice(shape_154, [0], full_int_array_579, full_int_array_580, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_155 = paddle._C_ops.shape(embedding_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_581 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_582 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_303 = paddle._C_ops.slice(shape_155, [0], full_int_array_581, full_int_array_582, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_956 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_957 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_958 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_344 = [full_956, slice_303, full_957]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_77 = paddle._C_ops.stack(combine_344, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_77 = paddle._C_ops.full_with_tensor(full_958, stack_77, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_959 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_960 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_961 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_345 = [full_959, slice_303, full_960]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_78 = paddle._C_ops.stack(combine_345, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_78 = paddle._C_ops.full_with_tensor(full_961, stack_78, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_231 = paddle._C_ops.transpose(embedding_38, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_346 = [full_with_tensor_77, full_with_tensor_78]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_347 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_962 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__152, rnn__153, rnn__154, rnn__155 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_231, combine_346, combine_347, None, full_962, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_232 = paddle._C_ops.transpose(rnn__152, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_233 = paddle._C_ops.transpose(transpose_232, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_963 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_964 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_348 = [slice_302, full_963, full_964]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_168, reshape_169 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_348]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_965 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_966 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_349 = [slice_302, full_965, full_966]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_170, reshape_171 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_349]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_234 = paddle._C_ops.transpose(transpose_233, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_150 = paddle._C_ops.matmul(transpose_234, reshape_168, False, False)

        # pd_op.full: (1xf32) <- ()
        full_967 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__39 = paddle._C_ops.scale_(matmul_150, full_967, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_156 = paddle._C_ops.shape(scale__39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_583 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_584 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_304 = paddle._C_ops.slice(shape_156, [0], full_int_array_583, full_int_array_584, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_968 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_969 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_970 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_350 = [slice_304, full_968, full_969, full_970]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__39, [x.reshape([]) for x in combine_350]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_157 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_585 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_586 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_305 = paddle._C_ops.slice(shape_157, [0], full_int_array_585, full_int_array_586, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_971 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_39 = paddle._C_ops.scale(slice_305, full_971, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_972 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_113 = paddle._C_ops.cast(scale_39, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_973 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_38 = paddle.arange(full_972, cast_113, full_973, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_974 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_114 = paddle._C_ops.cast(slice_305, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_38 = paddle._C_ops.memcpy_h2d(cast_114, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_38 = paddle._C_ops.less_than(full_974, memcpy_h2d_38)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_975 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_114 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_115 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_116 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xf32) <- (xb, xi64, -1x40x6x40xf32, xi64, -1x40x6x40xf32, xi32, xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12726 = 0
        while less_than_38:
            less_than_38, full_974, reshape__158, assign_value_116, full_975, assign_value_115, assign_value_114, = self.pd_op_while_12726_0_0(arange_38, feed_2, slice_305, less_than_38, full_974, reshape__158, assign_value_116, full_975, assign_value_115, assign_value_114)
            while_loop_counter_12726 += 1
            if while_loop_counter_12726 > kWhileLoopLimit:
                break
            
        while_228, while_229, while_230, while_231, while_232, while_233, = full_974, reshape__158, assign_value_116, full_975, assign_value_115, assign_value_114,

        # pd_op.full: (1xi32) <- ()
        full_976 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_977 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_351 = [slice_304, full_976, full_977]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_231, [x.reshape([]) for x in combine_351]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__75 = paddle._C_ops.softmax_(reshape__160, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_235 = paddle._C_ops.transpose(reshape_170, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_151 = paddle._C_ops.matmul(softmax__75, transpose_235, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_236 = paddle._C_ops.transpose(matmul_151, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_587 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_588 = [38]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_306 = paddle._C_ops.slice(transpose_236, [2], full_int_array_587, full_int_array_588, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_589 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_590 = [38]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_307 = paddle._C_ops.slice(transpose_8, [1], full_int_array_589, full_int_array_590, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_352 = [slice_306, slice_307]

        # pd_op.full: (1xi32) <- ()
        full_978 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_352, full_978)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_152 = paddle._C_ops.matmul(concat_37, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__94 = paddle._C_ops.add_(matmul_152, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_979 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_37 = paddle._C_ops.split_with_num(add__94, 2, full_979)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_308 = split_with_num_37[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__37 = paddle._C_ops.sigmoid_(slice_308)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_309 = split_with_num_37[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__37 = paddle._C_ops.multiply_(slice_309, sigmoid__37)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_153 = paddle._C_ops.matmul(multiply__37, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__95 = paddle._C_ops.add_(matmul_153, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__76 = paddle._C_ops.softmax_(add__95, -1)

        # pd_op.full: (1xi64) <- ()
        full_980 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_37 = paddle._C_ops.argmax(softmax__76, full_980, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_115 = paddle._C_ops.cast(argmax_37, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_591 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_592 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_593 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__37 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__36, cast_115, full_int_array_591, full_int_array_592, full_int_array_593, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_39 = paddle._C_ops.embedding(set_value_with_tensor__37, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_158 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_594 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_595 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_310 = paddle._C_ops.slice(shape_158, [0], full_int_array_594, full_int_array_595, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_159 = paddle._C_ops.shape(embedding_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_596 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_597 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_311 = paddle._C_ops.slice(shape_159, [0], full_int_array_596, full_int_array_597, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_981 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_982 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_983 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_353 = [full_981, slice_311, full_982]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_79 = paddle._C_ops.stack(combine_353, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_79 = paddle._C_ops.full_with_tensor(full_983, stack_79, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_984 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_985 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_986 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_354 = [full_984, slice_311, full_985]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_80 = paddle._C_ops.stack(combine_354, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_80 = paddle._C_ops.full_with_tensor(full_986, stack_80, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_237 = paddle._C_ops.transpose(embedding_39, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_355 = [full_with_tensor_79, full_with_tensor_80]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_356 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_987 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__156, rnn__157, rnn__158, rnn__159 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_237, combine_355, combine_356, None, full_987, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_238 = paddle._C_ops.transpose(rnn__156, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_239 = paddle._C_ops.transpose(transpose_238, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_988 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_989 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_357 = [slice_310, full_988, full_989]

        # pd_op.reshape: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_172, reshape_173 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__17, [x.reshape([]) for x in combine_357]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_990 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_991 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_358 = [slice_310, full_990, full_991]

        # pd_op.reshape: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape_174, reshape_175 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__27, [x.reshape([]) for x in combine_358]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_240 = paddle._C_ops.transpose(transpose_239, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_154 = paddle._C_ops.matmul(transpose_240, reshape_172, False, False)

        # pd_op.full: (1xf32) <- ()
        full_992 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__40 = paddle._C_ops.scale_(matmul_154, full_992, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_160 = paddle._C_ops.shape(scale__40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_598 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_599 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_312 = paddle._C_ops.slice(shape_160, [0], full_int_array_598, full_int_array_599, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_993 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_994 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_995 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_359 = [slice_312, full_993, full_994, full_995]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__40, [x.reshape([]) for x in combine_359]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_161 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_600 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_601 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_313 = paddle._C_ops.slice(shape_161, [0], full_int_array_600, full_int_array_601, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_996 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_40 = paddle._C_ops.scale(slice_313, full_996, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_997 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_116 = paddle._C_ops.cast(scale_40, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_998 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_39 = paddle.arange(full_997, cast_116, full_998, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_999 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_117 = paddle._C_ops.cast(slice_313, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_39 = paddle._C_ops.memcpy_h2d(cast_117, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_39 = paddle._C_ops.less_than(full_999, memcpy_h2d_39)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_1000 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_117 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_118 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_119 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi32, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64) <- (xb, xi32, xi64, xf32, -1x40x6x40xf32, -1x40x6x40xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_12880 = 0
        while less_than_39:
            less_than_39, assign_value_118, assign_value_119, assign_value_117, reshape__162, full_1000, full_999, = self.pd_op_while_12880_0_0(arange_39, feed_2, slice_313, less_than_39, assign_value_118, assign_value_119, assign_value_117, reshape__162, full_1000, full_999)
            while_loop_counter_12880 += 1
            if while_loop_counter_12880 > kWhileLoopLimit:
                break
            
        while_234, while_235, while_236, while_237, while_238, while_239, = assign_value_118, assign_value_119, assign_value_117, reshape__162, full_1000, full_999,

        # pd_op.full: (1xi32) <- ()
        full_1001 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1002 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_360 = [slice_312, full_1001, full_1002]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_238, [x.reshape([]) for x in combine_360]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__77 = paddle._C_ops.softmax_(reshape__164, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_241 = paddle._C_ops.transpose(reshape_174, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_155 = paddle._C_ops.matmul(softmax__77, transpose_241, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_242 = paddle._C_ops.transpose(matmul_155, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_602 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_603 = [39]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_314 = paddle._C_ops.slice(transpose_242, [2], full_int_array_602, full_int_array_603, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_604 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_605 = [39]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_315 = paddle._C_ops.slice(transpose_8, [1], full_int_array_604, full_int_array_605, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_361 = [slice_314, slice_315]

        # pd_op.full: (1xi32) <- ()
        full_1003 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_361, full_1003)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_156 = paddle._C_ops.matmul(concat_38, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__96 = paddle._C_ops.add_(matmul_156, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_1004 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_38 = paddle._C_ops.split_with_num(add__96, 2, full_1004)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_316 = split_with_num_38[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__38 = paddle._C_ops.sigmoid_(slice_316)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_317 = split_with_num_38[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__38 = paddle._C_ops.multiply_(slice_317, sigmoid__38)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_157 = paddle._C_ops.matmul(multiply__38, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__97 = paddle._C_ops.add_(matmul_157, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__78 = paddle._C_ops.softmax_(add__97, -1)

        # pd_op.full: (1xi64) <- ()
        full_1005 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi64) <- (-1x92xf32, 1xi64)
        argmax_38 = paddle._C_ops.argmax(softmax__78, full_1005, False, False, paddle.int64)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_118 = paddle._C_ops.cast(argmax_38, paddle.int64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_606 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_607 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_608 = [1]

        # pd_op.set_value_with_tensor_: (-1x40xi64) <- (-1x40xi64, -1xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__38 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__37, cast_118, full_int_array_606, full_int_array_607, full_int_array_608, [1], [1], [])

        # pd_op.embedding: (-1x40x128xf32) <- (-1x40xi64, 93x128xf32)
        embedding_40 = paddle._C_ops.embedding(set_value_with_tensor__38, parameter_171, 92, False)

        # pd_op.shape: (4xi32) <- (-1x128x6x40xf32)
        shape_162 = paddle._C_ops.shape(add__17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_609 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_610 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_318 = paddle._C_ops.slice(shape_162, [0], full_int_array_609, full_int_array_610, [1], [0])

        # pd_op.shape: (3xi32) <- (-1x40x128xf32)
        shape_163 = paddle._C_ops.shape(embedding_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_611 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_612 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_319 = paddle._C_ops.slice(shape_163, [0], full_int_array_611, full_int_array_612, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_1006 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_1007 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_1008 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_362 = [full_1006, slice_319, full_1007]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_81 = paddle._C_ops.stack(combine_362, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_81 = paddle._C_ops.full_with_tensor(full_1008, stack_81, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_1009 = paddle._C_ops.full([], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_1010 = paddle._C_ops.full([], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_1011 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_363 = [full_1009, slice_319, full_1010]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_82 = paddle._C_ops.stack(combine_363, 0)

        # pd_op.full_with_tensor: (2x-1x128xf32) <- (1xf32, 3xi32)
        full_with_tensor_82 = paddle._C_ops.full_with_tensor(full_1011, stack_82, paddle.float32)

        # pd_op.transpose: (40x-1x128xf32) <- (-1x40x128xf32)
        transpose_243 = paddle._C_ops.transpose(embedding_40, [1, 0, 2])

        # builtin.combine: ([2x-1x128xf32, 2x-1x128xf32]) <- (2x-1x128xf32, 2x-1x128xf32)
        combine_364 = [full_with_tensor_81, full_with_tensor_82]

        # builtin.combine: ([512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32]) <- (512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        combine_365 = [parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179]

        # pd_op.full: (xui8) <- ()
        full_1012 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (40x-1x128xf32, xui8, [2x-1x128xf32, 2x-1x128xf32], xui8) <- (40x-1x128xf32, [2x-1x128xf32, 2x-1x128xf32], [512x128xf32, 512x128xf32, 512x128xf32, 512x128xf32, 512xf32, 512xf32, 512xf32, 512xf32], None, xui8)
        rnn__160, rnn__161, rnn__162, rnn__163 = (lambda x, f: f(x))(paddle._C_ops.rnn(transpose_243, combine_364, combine_365, None, full_1012, float('0'), False, 128, 128, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.transpose: (-1x40x128xf32) <- (40x-1x128xf32)
        transpose_244 = paddle._C_ops.transpose(rnn__160, [1, 0, 2])

        # pd_op.transpose: (-1x128x40xf32) <- (-1x40x128xf32)
        transpose_245 = paddle._C_ops.transpose(transpose_244, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_1013 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1014 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_366 = [slice_318, full_1013, full_1014]

        # pd_op.reshape_: (-1x128x240xf32, 0x-1x128x6x40xf32) <- (-1x128x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__17, [x.reshape([]) for x in combine_366]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_1015 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1016 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_367 = [slice_318, full_1015, full_1016]

        # pd_op.reshape_: (-1x512x240xf32, 0x-1x512x6x40xf32) <- (-1x512x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(relu__27, [x.reshape([]) for x in combine_367]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x128xf32) <- (-1x128x40xf32)
        transpose_246 = paddle._C_ops.transpose(transpose_245, [0, 2, 1])

        # pd_op.matmul: (-1x40x240xf32) <- (-1x40x128xf32, -1x128x240xf32)
        matmul_158 = paddle._C_ops.matmul(transpose_246, reshape__166, False, False)

        # pd_op.full: (1xf32) <- ()
        full_1017 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x240xf32) <- (-1x40x240xf32, 1xf32)
        scale__41 = paddle._C_ops.scale_(matmul_158, full_1017, float('0'), True)

        # pd_op.shape: (3xi32) <- (-1x40x240xf32)
        shape_164 = paddle._C_ops.shape(scale__41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_613 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_614 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_320 = paddle._C_ops.slice(shape_164, [0], full_int_array_613, full_int_array_614, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_1018 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1019 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1020 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32, 1xi32)
        combine_368 = [slice_320, full_1018, full_1019, full_1020]

        # pd_op.reshape_: (-1x40x6x40xf32, 0x-1x40x240xf32) <- (-1x40x240xf32, [xi32, 1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(scale__41, [x.reshape([]) for x in combine_368]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (1xi32) <- (-1xf32)
        shape_165 = paddle._C_ops.shape(feed_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_615 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_616 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_321 = paddle._C_ops.slice(shape_165, [0], full_int_array_615, full_int_array_616, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_1021 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_41 = paddle._C_ops.scale(slice_321, full_1021, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_1022 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_119 = paddle._C_ops.cast(scale_41, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_1023 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_40 = paddle.arange(full_1022, cast_119, full_1023, dtype='int64')

        # pd_op.full: (xi64) <- ()
        full_1024 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.cast: (xi64) <- (xi32)
        cast_120 = paddle._C_ops.cast(slice_321, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_40 = paddle._C_ops.memcpy_h2d(cast_120, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_40 = paddle._C_ops.less_than(full_1024, memcpy_h2d_40)

        # pd_op.full: (-1x40x6x40xf32) <- ()
        full_1025 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (xf32) <- ()
        assign_value_120 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi32) <- ()
        assign_value_121 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_122 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (xi64, xi32, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32) <- (xb, xi64, xi32, xi64, -1x40x6x40xf32, xf32, -1x40x6x40xf32)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_13034 = 0
        while less_than_40:
            less_than_40, full_1024, assign_value_121, assign_value_122, full_1025, assign_value_120, reshape__170, = self.pd_op_while_13034_0_0(arange_40, feed_2, slice_321, less_than_40, full_1024, assign_value_121, assign_value_122, full_1025, assign_value_120, reshape__170)
            while_loop_counter_13034 += 1
            if while_loop_counter_13034 > kWhileLoopLimit:
                break
            
        while_240, while_241, while_242, while_243, while_244, while_245, = full_1024, assign_value_121, assign_value_122, full_1025, assign_value_120, reshape__170,

        # pd_op.full: (1xi32) <- ()
        full_1026 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1027 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_369 = [slice_320, full_1026, full_1027]

        # pd_op.reshape_: (-1x40x240xf32, 0x-1x40x6x40xf32) <- (-1x40x6x40xf32, [xi32, 1xi32, 1xi32])
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(while_243, [x.reshape([]) for x in combine_369]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x40x240xf32) <- (-1x40x240xf32)
        softmax__79 = paddle._C_ops.softmax_(reshape__172, 2)

        # pd_op.transpose: (-1x240x512xf32) <- (-1x512x240xf32)
        transpose_247 = paddle._C_ops.transpose(reshape__168, [0, 2, 1])

        # pd_op.matmul: (-1x40x512xf32) <- (-1x40x240xf32, -1x240x512xf32)
        matmul_159 = paddle._C_ops.matmul(softmax__79, transpose_247, False, False)

        # pd_op.transpose: (-1x512x40xf32) <- (-1x40x512xf32)
        transpose_248 = paddle._C_ops.transpose(matmul_159, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_617 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_618 = [40]

        # pd_op.slice: (-1x512xf32) <- (-1x512x40xf32, 1xi64, 1xi64)
        slice_322 = paddle._C_ops.slice(transpose_248, [2], full_int_array_617, full_int_array_618, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_619 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_620 = [40]

        # pd_op.slice: (-1x512xf32) <- (-1x40x512xf32, 1xi64, 1xi64)
        slice_323 = paddle._C_ops.slice(transpose_8, [1], full_int_array_619, full_int_array_620, [1], [1])

        # builtin.combine: ([-1x512xf32, -1x512xf32]) <- (-1x512xf32, -1x512xf32)
        combine_370 = [slice_322, slice_323]

        # pd_op.full: (1xi32) <- ()
        full_1028 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024xf32) <- ([-1x512xf32, -1x512xf32], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_370, full_1028)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_160 = paddle._C_ops.matmul(concat_39, parameter_180, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__98 = paddle._C_ops.add_(matmul_160, parameter_181)

        # pd_op.full: (1xi32) <- ()
        full_1029 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512xf32, -1x512xf32]) <- (-1x1024xf32, 1xi32)
        split_with_num_39 = paddle._C_ops.split_with_num(add__98, 2, full_1029)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_324 = split_with_num_39[1]

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__39 = paddle._C_ops.sigmoid_(slice_324)

        # builtin.slice: (-1x512xf32) <- ([-1x512xf32, -1x512xf32])
        slice_325 = split_with_num_39[0]

        # pd_op.multiply_: (-1x512xf32) <- (-1x512xf32, -1x512xf32)
        multiply__39 = paddle._C_ops.multiply_(slice_325, sigmoid__39)

        # pd_op.matmul: (-1x92xf32) <- (-1x512xf32, 512x92xf32)
        matmul_161 = paddle._C_ops.matmul(multiply__39, parameter_182, False, False)

        # pd_op.add_: (-1x92xf32) <- (-1x92xf32, 92xf32)
        add__99 = paddle._C_ops.add_(matmul_161, parameter_183)

        # pd_op.softmax_: (-1x92xf32) <- (-1x92xf32)
        softmax__80 = paddle._C_ops.softmax_(add__99, -1)

        # builtin.combine: ([-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32]) <- (-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32)
        combine_371 = [softmax__2, softmax__4, softmax__6, softmax__8, softmax__10, softmax__12, softmax__14, softmax__16, softmax__18, softmax__20, softmax__22, softmax__24, softmax__26, softmax__28, softmax__30, softmax__32, softmax__34, softmax__36, softmax__38, softmax__40, softmax__42, softmax__44, softmax__46, softmax__48, softmax__50, softmax__52, softmax__54, softmax__56, softmax__58, softmax__60, softmax__62, softmax__64, softmax__66, softmax__68, softmax__70, softmax__72, softmax__74, softmax__76, softmax__78, softmax__80]

        # pd_op.stack: (-1x40x92xf32) <- ([-1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32, -1x92xf32])
        stack_83 = paddle._C_ops.stack(combine_371, 1)

        # pd_op.full: (1xf32) <- ()
        full_1030 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x40x92xf32) <- (-1x40x92xf32, 1xf32)
        scale__42 = paddle._C_ops.scale_(stack_83, full_1030, float('0'), True)
        return scale__42



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

    def forward(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_28, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, feed_1, feed_2, feed_0):
        return self.builtin_module_6478_0_0(parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_28, parameter_32, parameter_29, parameter_31, parameter_30, parameter_33, parameter_37, parameter_34, parameter_36, parameter_35, parameter_38, parameter_42, parameter_39, parameter_41, parameter_40, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, feed_1, feed_2, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_6478_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([41, 128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([93, 128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512, 92], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([92], dtype='float32', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([1, 40]),
            # feed_2
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            # feed_0
            paddle.uniform([1, 3, 48, 160], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[41, 128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[93, 128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512, 92], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[92], dtype='float32'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 40], dtype='int64'),
            # feed_2
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 48, 160], dtype='float32'),
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