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
    return [73, 166][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_while_776_0_0(self, arange_0, transpose_0, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, slice_7, less_than_1, full_with_tensor_0, assign_value_0, full_with_tensor_1, full_9, full_with_tensor_2, full_1, assign_value_2, assign_value_1, assign_value_4, assign_value_3):

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

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('30'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.one_hot: (-1x30xf32) <- (-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(full_with_tensor_0 % paddle.cast(full_2, full_with_tensor_0.dtype), full_2)

        # pd_op.matmul: (-1x256x256xf32) <- (-1x256x576xf32, 576x256xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_0, parameter_0, False, False)

        # pd_op.matmul: (-1x256xf32) <- (-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(full_with_tensor_1, parameter_1, False, False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__0 = paddle._C_ops.add_(matmul_1, parameter_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze_: (-1x1x256xf32, None) <- (-1x256xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(add__0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x256xf32) <- (-1x256x256xf32, -1x1x256xf32)
        add__1 = paddle._C_ops.add_(matmul_0, unsqueeze__0)

        # pd_op.tanh_: (-1x256x256xf32) <- (-1x256x256xf32)
        tanh__0 = paddle._C_ops.tanh_(add__1)

        # pd_op.matmul: (-1x256x1xf32) <- (-1x256x256xf32, 256x1xf32)
        matmul_2 = paddle._C_ops.matmul(tanh__0, parameter_3, False, False)

        # pd_op.softmax_: (-1x256x1xf32) <- (-1x256x1xf32)
        softmax__0 = paddle._C_ops.softmax_(matmul_2, 1)

        # pd_op.transpose: (-1x1x256xf32) <- (-1x256x1xf32)
        transpose_1 = paddle._C_ops.transpose(softmax__0, [0, 2, 1])

        # pd_op.matmul: (-1x1x576xf32) <- (-1x1x256xf32, -1x256x576xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_1, transpose_0, False, False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.squeeze_: (-1x576xf32, None) <- (-1x1x576xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(matmul_3, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x576xf32, -1x30xf32]) <- (-1x576xf32, -1x30xf32)
        combine_2 = [squeeze__0, one_hot_0]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x606xf32) <- ([-1x576xf32, -1x30xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_3)

        # pd_op.matmul: (-1x768xf32) <- (-1x606xf32, 768x606xf32)
        matmul_4 = paddle._C_ops.matmul(concat_0, parameter_4, False, True)

        # pd_op.add_: (-1x768xf32) <- (-1x768xf32, 768xf32)
        add__2 = paddle._C_ops.add_(matmul_4, parameter_5)

        # pd_op.matmul: (-1x768xf32) <- (-1x256xf32, 768x256xf32)
        matmul_5 = paddle._C_ops.matmul(full_with_tensor_1, parameter_6, False, True)

        # pd_op.add_: (-1x768xf32) <- (-1x768xf32, 768xf32)
        add__3 = paddle._C_ops.add_(matmul_5, parameter_7)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x768xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add__2, 3, full_4)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256xf32, -1x256xf32, -1x256xf32]) <- (-1x768xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__3, 3, full_5)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_1 = split_with_num_0[0]

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_2 = split_with_num_1[0]

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__4 = paddle._C_ops.add_(slice_1, slice_2)

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__4)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_3 = split_with_num_0[1]

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_4 = split_with_num_1[1]

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__5 = paddle._C_ops.add_(slice_3, slice_4)

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__5)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_5 = split_with_num_1[2]

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__0 = paddle._C_ops.multiply_(sigmoid__0, slice_5)

        # builtin.slice: (-1x256xf32) <- ([-1x256xf32, -1x256xf32, -1x256xf32])
        slice_6 = split_with_num_0[2]

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__6 = paddle._C_ops.add_(slice_6, multiply__0)

        # pd_op.tanh_: (-1x256xf32) <- (-1x256xf32)
        tanh__1 = paddle._C_ops.tanh_(add__6)

        # pd_op.subtract: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        subtract_0 = paddle._C_ops.subtract(full_with_tensor_1, tanh__1)

        # pd_op.multiply_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        multiply__1 = paddle._C_ops.multiply_(subtract_0, sigmoid__1)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        add__7 = paddle._C_ops.add_(multiply__1, tanh__1)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_6, float('1'), True)

        # pd_op.cast: (-1x256xf32) <- (-1x256xf32)
        cast_0 = paddle._C_ops.cast(add__7, paddle.float32)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [slice_0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_4 = [scale_1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.set_value_with_tensor: (-1x501x256xf32) <- (-1x501x256xf32, -1x256xf32, [xi64], [xi64], 1xi64)
        set_value_with_tensor_0 = paddle._C_ops.set_value_with_tensor(full_with_tensor_2, cast_0, [x.reshape([]) for x in combine_3], [x.reshape([]) for x in combine_4], full_int_array_2, [1], [1], [])

        # pd_op.matmul: (-1x30xf32) <- (-1x256xf32, 256x30xf32)
        matmul_6 = paddle._C_ops.matmul(add__7, parameter_8, False, False)

        # pd_op.add_: (-1x30xf32) <- (-1x30xf32, 30xf32)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_9)

        # pd_op.full: (1xi64) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1xi32) <- (-1x30xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(add__8, full_7, False, False, paddle.int32)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(full_1, full_8, float('1'), True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_7, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(scale_2, memcpy_h2d_0)

        # pd_op.assign_out_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        assign_out__0 = paddle._C_ops.assign_out_(add__7, full_with_tensor_1)

        # pd_op.assign_out_: (-1x30xf32) <- (-1x30xf32, -1x30xf32)
        assign_out__1 = paddle._C_ops.assign_out_(one_hot_0, assign_value_0)

        # pd_op.assign_out_: (-1x501x256xf32) <- (-1x501x256xf32, -1x501x256xf32)
        assign_out__2 = paddle._C_ops.assign_out_(set_value_with_tensor_0, full_with_tensor_2)

        # pd_op.assign_out_: (-1xi32) <- (-1xi32, -1xi32)
        assign_out__3 = paddle._C_ops.assign_out_(argmax_0, full_with_tensor_0)

        # pd_op.assign_out_: (-1x30xf32) <- (-1x30xf32, -1x30xf32)
        assign_out__4 = paddle._C_ops.assign_out_(add__8, assign_value_1)

        # pd_op.assign_out_: (-1x1x256xf32) <- (-1x1x256xf32, -1x1x256xf32)
        assign_out__5 = paddle._C_ops.assign_out_(transpose_1, assign_value_2)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__6 = paddle._C_ops.assign_out_(scale_2, full_1)

        # pd_op.assign_out_: (xi64) <- (xi64, xi64)
        assign_out__7 = paddle._C_ops.assign_out_(slice_0, assign_value_3)

        # pd_op.assign_out_: (-1x256xf32) <- (-1x256xf32, -1x256xf32)
        assign_out__8 = paddle._C_ops.assign_out_(add__7, assign_value_4)

        # pd_op.assign_out_: (xb) <- (xb, xb)
        assign_out__9 = paddle._C_ops.assign_out_(less_than_0, less_than_1)
        return assign_out__9, assign_out__3, assign_out__1, assign_out__0, set_value_with_tensor_0, assign_out__2, assign_out__6, assign_out__5, assign_out__4, assign_out__8, assign_out__7
    def builtin_module_436_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_184, parameter_179, parameter_177, parameter_181, parameter_183, parameter_176, parameter_182, parameter_178, parameter_180, parameter_175, parameter_185, parameter_186, parameter_187, parameter_188, feed_0):

        # pd_op.conv2d: (-1x16x244x244xf32) <- (-1x3x488x488xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x244x244xf32, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x244x244xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x244x244xf32) <- (-1x16x244x244xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (-1x16x244x244xf32) <- (-1x16x244x244xf32, 16x16x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(hardswish_0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x244x244xf32, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x244x244xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x244x244xf32) <- (-1x16x244x244xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.depthwise_conv2d: (-1x16x122x122xf32) <- (-1x16x244x244xf32, 16x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(relu__0, parameter_10, [2, 2], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x122x122xf32, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x122x122xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x122x122xf32) <- (-1x16x122x122xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x16x122x122xf32) <- (-1x16x122x122xf32, 16x16x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x122x122xf32, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x122x122xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x72x122x122xf32) <- (-1x16x122x122xf32, 72x16x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__18, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x122x122xf32, 72xf32, 72xf32, 72xf32, 72xf32, None) <- (-1x72x122x122xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x72x122x122xf32) <- (-1x72x122x122xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.depthwise_conv2d: (-1x72x61x61xf32) <- (-1x72x122x122xf32, 72x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(relu__2, parameter_25, [2, 2], [1, 1], 'EXPLICIT', 72, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x72x61x61xf32, 72xf32, 72xf32, 72xf32, 72xf32, None) <- (-1x72x61x61xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x72x61x61xf32) <- (-1x72x61x61xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x24x61x61xf32) <- (-1x72x61x61xf32, 24x72x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x61x61xf32, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x61x61xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x88x61x61xf32) <- (-1x24x61x61xf32, 88x24x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x88x61x61xf32, 88xf32, 88xf32, 88xf32, 88xf32, None) <- (-1x88x61x61xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x88x61x61xf32) <- (-1x88x61x61xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.depthwise_conv2d: (-1x88x61x61xf32) <- (-1x88x61x61xf32, 88x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(relu__4, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 88, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x88x61x61xf32, 88xf32, 88xf32, 88xf32, 88xf32, None) <- (-1x88x61x61xf32, 88xf32, 88xf32, 88xf32, 88xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x88x61x61xf32) <- (-1x88x61x61xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x24x61x61xf32) <- (-1x88x61x61xf32, 24x88x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x61x61xf32, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x61x61xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x24x61x61xf32) <- (-1x24x61x61xf32, -1x24x61x61xf32)
        add__0 = paddle._C_ops.add_(batch_norm__36, batch_norm__54)

        # pd_op.conv2d: (-1x96x61x61xf32) <- (-1x24x61x61xf32, 96x24x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(add__0, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x61x61xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x61x61xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x61x61xf32) <- (-1x96x61x61xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x96x31x31xf32) <- (-1x96x61x61xf32, 96x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(hardswish_1, parameter_55, [2, 2], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x31x31xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x31x31xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x31x31xf32) <- (-1x96x31x31xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__66)

        # pd_op.conv2d: (-1x40x31x31xf32) <- (-1x96x31x31xf32, 40x96x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(hardswish_2, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32, None) <- (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x240x31x31xf32) <- (-1x40x31x31xf32, 240x40x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(batch_norm__72, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x240x31x31xf32) <- (-1x240x31x31xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__78)

        # pd_op.depthwise_conv2d: (-1x240x31x31xf32) <- (-1x240x31x31xf32, 240x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(hardswish_3, parameter_70, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x240x31x31xf32) <- (-1x240x31x31xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__84)

        # pd_op.conv2d: (-1x40x31x31xf32) <- (-1x240x31x31xf32, 40x240x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(hardswish_4, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32, None) <- (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x31x31xf32) <- (-1x40x31x31xf32, -1x40x31x31xf32)
        add__1 = paddle._C_ops.add_(batch_norm__72, batch_norm__90)

        # pd_op.conv2d: (-1x240x31x31xf32) <- (-1x40x31x31xf32, 240x40x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(add__1, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x240x31x31xf32) <- (-1x240x31x31xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__96)

        # pd_op.depthwise_conv2d: (-1x240x31x31xf32) <- (-1x240x31x31xf32, 240x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(hardswish_5, parameter_85, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32, None) <- (-1x240x31x31xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x240x31x31xf32) <- (-1x240x31x31xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__102)

        # pd_op.conv2d: (-1x40x31x31xf32) <- (-1x240x31x31xf32, 40x240x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(hardswish_6, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32, None) <- (-1x40x31x31xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x40x31x31xf32) <- (-1x40x31x31xf32, -1x40x31x31xf32)
        add__2 = paddle._C_ops.add_(add__1, batch_norm__108)

        # pd_op.conv2d: (-1x120x31x31xf32) <- (-1x40x31x31xf32, 120x40x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(add__2, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x120x31x31xf32, 120xf32, 120xf32, 120xf32, 120xf32, None) <- (-1x120x31x31xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x120x31x31xf32) <- (-1x120x31x31xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.depthwise_conv2d: (-1x120x31x31xf32) <- (-1x120x31x31xf32, 120x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(hardswish_7, parameter_100, [1, 1], [2, 2], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x120x31x31xf32, 120xf32, 120xf32, 120xf32, 120xf32, None) <- (-1x120x31x31xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x120x31x31xf32) <- (-1x120x31x31xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.conv2d: (-1x48x31x31xf32) <- (-1x120x31x31xf32, 48x120x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(hardswish_8, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x31x31xf32, 48xf32, 48xf32, 48xf32, 48xf32, None) <- (-1x48x31x31xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x144x31x31xf32) <- (-1x48x31x31xf32, 144x48x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__126, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x31x31xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x31x31xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x144x31x31xf32) <- (-1x144x31x31xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__132)

        # pd_op.depthwise_conv2d: (-1x144x31x31xf32) <- (-1x144x31x31xf32, 144x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(hardswish_9, parameter_115, [1, 1], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x144x31x31xf32, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x31x31xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x144x31x31xf32) <- (-1x144x31x31xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__138)

        # pd_op.conv2d: (-1x48x31x31xf32) <- (-1x144x31x31xf32, 48x144x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(hardswish_10, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x31x31xf32, 48xf32, 48xf32, 48xf32, 48xf32, None) <- (-1x48x31x31xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x48x31x31xf32) <- (-1x48x31x31xf32, -1x48x31x31xf32)
        add__3 = paddle._C_ops.add_(batch_norm__126, batch_norm__144)

        # pd_op.conv2d: (-1x288x31x31xf32) <- (-1x48x31x31xf32, 288x48x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(add__3, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x288x31x31xf32, 288xf32, 288xf32, 288xf32, 288xf32, None) <- (-1x288x31x31xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x288x31x31xf32) <- (-1x288x31x31xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__150)

        # pd_op.depthwise_conv2d: (-1x288x16x16xf32) <- (-1x288x31x31xf32, 288x1x5x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(hardswish_11, parameter_130, [2, 2], [2, 2], 'EXPLICIT', 288, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x288x16x16xf32, 288xf32, 288xf32, 288xf32, 288xf32, None) <- (-1x288x16x16xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x288x16x16xf32) <- (-1x288x16x16xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__156)

        # pd_op.conv2d: (-1x96x16x16xf32) <- (-1x288x16x16xf32, 96x288x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(hardswish_12, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x576x16x16xf32) <- (-1x96x16x16xf32, 576x96x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__162, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x576x16x16xf32) <- (-1x576x16x16xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__168)

        # pd_op.depthwise_conv2d: (-1x576x16x16xf32) <- (-1x576x16x16xf32, 576x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(hardswish_13, parameter_145, [1, 1], [2, 2], 'EXPLICIT', 576, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x576x16x16xf32) <- (-1x576x16x16xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__174)

        # pd_op.conv2d: (-1x96x16x16xf32) <- (-1x576x16x16xf32, 96x576x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(hardswish_14, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x96x16x16xf32) <- (-1x96x16x16xf32, -1x96x16x16xf32)
        add__4 = paddle._C_ops.add_(batch_norm__162, batch_norm__180)

        # pd_op.conv2d: (-1x576x16x16xf32) <- (-1x96x16x16xf32, 576x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(add__4, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x576x16x16xf32) <- (-1x576x16x16xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__186)

        # pd_op.depthwise_conv2d: (-1x576x16x16xf32) <- (-1x576x16x16xf32, 576x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(hardswish_15, parameter_160, [1, 1], [2, 2], 'EXPLICIT', 576, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x576x16x16xf32) <- (-1x576x16x16xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__192)

        # pd_op.conv2d: (-1x96x16x16xf32) <- (-1x576x16x16xf32, 96x576x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(hardswish_16, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32, None) <- (-1x96x16x16xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x96x16x16xf32) <- (-1x96x16x16xf32, -1x96x16x16xf32)
        add__5 = paddle._C_ops.add_(add__4, batch_norm__198)

        # pd_op.conv2d: (-1x576x16x16xf32) <- (-1x96x16x16xf32, 576x96x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(add__5, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x16x16xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x576x16x16xf32) <- (-1x576x16x16xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__204)

        # pd_op.shape: (4xi32) <- (-1x576x16x16xf32)
        shape_0 = paddle._C_ops.shape(hardswish_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, 1xi32, 1xi32]) <- (xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1]

        # pd_op.reshape_: (-1x576x256xf32, 0x-1x576x16x16xf32) <- (-1x576x16x16xf32, [xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(hardswish_17, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x576xf32) <- (-1x576x256xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x256x576xf32)
        shape_1 = paddle._C_ops.shape(transpose_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full: (xi32) <- ()
        full_2 = paddle._C_ops.full([], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32]) <- (xi32, xi32)
        combine_1 = [slice_1, full_2]

        # pd_op.stack: (2xi32) <- ([xi32, xi32])
        stack_0 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.full_with_tensor: (-1x256xf32) <- (1xf32, 2xi32)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_3, stack_0, paddle.float32)

        # pd_op.full: (xi32) <- ()
        full_4 = paddle._C_ops.full([], float('501'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full([], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32, xi32, xi32]) <- (xi32, xi32, xi32)
        combine_2 = [slice_1, full_4, full_5]

        # pd_op.stack: (3xi32) <- ([xi32, xi32, xi32])
        stack_1 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full_with_tensor: (-1x501x256xf32) <- (1xf32, 3xi32)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_6, stack_1, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # builtin.combine: ([xi32]) <- (xi32)
        combine_3 = [slice_1]

        # pd_op.stack: (1xi32) <- ([xi32])
        stack_2 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.full_with_tensor: (-1xi32) <- (1xf32, 1xi32)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_7, stack_2, paddle.int32)

        # pd_op.assign_value: (xi64) <- ()
        assign_value_0 = paddle.to_tensor([500], dtype=paddle.int64).reshape([])

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (xi64) <- (xi64, 1xf32)
        scale__0 = paddle._C_ops.scale_(assign_value_0, full_8, float('1'), True)

        # pd_op.full: (1xi64) <- ()
        full_9 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_9, scale__0, full_10, dtype='int64')

        # pd_op.shape: (1xi32) <- (-1xi64)
        shape_2 = paddle._C_ops.shape(arange_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.assign_value: (-1x30xf32) <- ()
        assign_value_1 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (-1x1x256xf32) <- ()
        assign_value_2 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full([], float('0'), paddle.int64, paddle.framework._current_expected_place())

        # pd_op.assign_value: (-1x256xf32) <- ()
        assign_value_3 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.memcpy_h2d: (xi64) <- (xi64)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_0, 1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(full_11, memcpy_h2d_0)

        # pd_op.full: (-1x501x256xf32) <- ()
        full_12 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign_value: (-1x30xf32) <- ()
        assign_value_4 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.assign_value: (xi64) <- ()
        assign_value_5 = paddle.to_tensor([float('1.77113e+27')], dtype=paddle.float64).reshape([1])

        # pd_op.while: (-1xi32, -1x30xf32, -1x256xf32, -1x501x256xf32, -1x501x256xf32, xi64, -1x1x256xf32, -1x30xf32, -1x256xf32, xi64) <- (xb, -1xi32, -1x30xf32, -1x256xf32, -1x501x256xf32, -1x501x256xf32, xi64, -1x1x256xf32, -1x30xf32, -1x256xf32, xi64)
        import os
        ATHENA_WHILE_LOOP_LIMIT = os.getenv('ATHENA_WHILE_LOOP_LIMIT')
        kWhileLoopLimit = (128 if ATHENA_WHILE_LOOP_LIMIT is None else int(ATHENA_WHILE_LOOP_LIMIT))
        while_loop_counter_776 = 0
        while less_than_0:
            less_than_0, full_with_tensor_2, assign_value_1, full_with_tensor_0, full_12, full_with_tensor_1, full_11, assign_value_2, assign_value_4, assign_value_3, assign_value_5, = self.pd_op_while_776_0_0(arange_0, transpose_0, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, slice_2, less_than_0, full_with_tensor_2, assign_value_1, full_with_tensor_0, full_12, full_with_tensor_1, full_11, assign_value_2, assign_value_4, assign_value_3, assign_value_5)
            while_loop_counter_776 += 1
            if while_loop_counter_776 > kWhileLoopLimit:
                break
            
        while_0, while_1, while_2, while_3, while_4, while_5, while_6, while_7, while_8, while_9, = full_with_tensor_2, assign_value_1, full_with_tensor_0, full_12, full_with_tensor_1, full_11, assign_value_2, assign_value_4, assign_value_3, assign_value_5,

        # pd_op.matmul: (-1x501x30xf32) <- (-1x501x256xf32, 256x30xf32)
        matmul_0 = paddle._C_ops.matmul(while_3, parameter_183, False, False)

        # pd_op.add_: (-1x501x30xf32) <- (-1x501x30xf32, 30xf32)
        add__6 = paddle._C_ops.add_(matmul_0, parameter_184)

        # pd_op.softmax_: (-1x501x30xf32) <- (-1x501x30xf32)
        softmax__0 = paddle._C_ops.softmax_(add__6, -1)

        # pd_op.transpose: (-1x576x256xf32) <- (-1x256x576xf32)
        transpose_1 = paddle._C_ops.transpose(transpose_0, [0, 2, 1])

        # pd_op.matmul: (-1x576x501xf32) <- (-1x576x256xf32, 256x501xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_1, parameter_185, False, False)

        # pd_op.add_: (-1x576x501xf32) <- (-1x576x501xf32, 501xf32)
        add__7 = paddle._C_ops.add_(matmul_1, parameter_186)

        # pd_op.transpose: (-1x501x576xf32) <- (-1x576x501xf32)
        transpose_2 = paddle._C_ops.transpose(add__7, [0, 2, 1])

        # builtin.combine: ([-1x501x256xf32, -1x501x576xf32]) <- (-1x501x256xf32, -1x501x576xf32)
        combine_4 = [while_3, transpose_2]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x501x832xf32) <- ([-1x501x256xf32, -1x501x576xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_4, full_13)

        # pd_op.matmul: (-1x501x4xf32) <- (-1x501x832xf32, 832x4xf32)
        matmul_2 = paddle._C_ops.matmul(concat_0, parameter_187, False, False)

        # pd_op.add_: (-1x501x4xf32) <- (-1x501x4xf32, 4xf32)
        add__8 = paddle._C_ops.add_(matmul_2, parameter_188)

        # pd_op.sigmoid_: (-1x501x4xf32) <- (-1x501x4xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__8)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x501x4xf32) <- (-1x501x4xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(sigmoid__0, full_14, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x501x30xf32) <- (-1x501x30xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(softmax__0, full_15, float('0'), True)
        return scale__1, scale__2



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_184, parameter_179, parameter_177, parameter_181, parameter_183, parameter_176, parameter_182, parameter_178, parameter_180, parameter_175, parameter_185, parameter_186, parameter_187, parameter_188, feed_0):
        return self.builtin_module_436_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_184, parameter_179, parameter_177, parameter_181, parameter_183, parameter_176, parameter_182, parameter_178, parameter_180, parameter_175, parameter_185, parameter_186, parameter_187, parameter_188, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_436_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([16, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([16, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([72, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([72, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([24, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([88, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([88, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([88], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([24, 88, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([40, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([40, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([120, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([120, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([48, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([144, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([48, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([288, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([288, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([96, 288, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([576, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([576, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([96, 576, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([576, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([576, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([96, 576, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([576, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([768, 606], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([768, 256], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256, 30], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([576, 256], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([256, 501], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([501], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([832, 4], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 488, 488], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[72, 16, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[72, 1, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[24, 72, 1, 1], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[88, 24, 1, 1], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[88, 1, 3, 3], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[88], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[24, 88, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[40, 96, 1, 1], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[240, 40, 1, 1], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[40, 240, 1, 1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[120, 40, 1, 1], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[120, 1, 5, 5], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[48, 120, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[144, 48, 1, 1], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[48, 144, 1, 1], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[288, 48, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[288, 1, 5, 5], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[576, 96, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[576, 1, 5, 5], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[96, 576, 1, 1], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[576, 96, 1, 1], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[576, 1, 5, 5], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[96, 576, 1, 1], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[576, 96, 1, 1], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[768, 606], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[768, 256], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256, 30], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256, 1], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[576, 256], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[256, 501], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[501], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[832, 4], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 488, 488], dtype='float32'),
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