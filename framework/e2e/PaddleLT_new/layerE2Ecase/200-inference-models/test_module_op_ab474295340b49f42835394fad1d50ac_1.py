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
    return [437][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_596_0_0(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_26, parameter_27, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_55, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_62, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_89, parameter_88, parameter_90, parameter_91, parameter_92, parameter_93, parameter_95, parameter_94, parameter_96, parameter_97, parameter_98, parameter_99, parameter_101, parameter_100, parameter_102, parameter_103, parameter_104, parameter_105, parameter_107, parameter_106, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_125, parameter_124, parameter_126, parameter_127, parameter_128, parameter_129, parameter_131, parameter_130, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(cast_0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.conv2d: (-1x768x14x14xf16) <- (-1x3x224x224xf16, 768x3x16x16xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [16, 16], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf16, 0x768xf16) <- (768xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x14x14xf16) <- (-1x768x14x14xf16, 1x768x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.flatten_: (-1x768x196xf16, None) <- (-1x768x14x14xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x768xf16) <- (-1x768x196xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1]

        # pd_op.expand: (-1x1x768xf16) <- (1x1x768xf16, [1xi32, 1xi32, 1xi32])
        expand_0 = paddle._C_ops.expand(parameter_2, [x.reshape([]) for x in combine_0])

        # pd_op.cast: (-1x1x768xf32) <- (-1x1x768xf16)
        cast_1 = paddle._C_ops.cast(expand_0, paddle.float32)

        # pd_op.cast: (-1x1x768xf16) <- (-1x1x768xf32)
        cast_2 = paddle._C_ops.cast(cast_1, paddle.float16)

        # builtin.combine: ([-1x1x768xf16, -1x196x768xf16]) <- (-1x1x768xf16, -1x196x768xf16)
        combine_1 = [cast_2, transpose_0]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x197x768xf16) <- ([-1x1x768xf16, -1x196x768xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 1x197x768xf16)
        add__1 = paddle._C_ops.add_(concat_0, parameter_3)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__1, parameter_4, parameter_5, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_6, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__2 = paddle._C_ops.add_(matmul_0, parameter_7)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_3 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(transpose_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_2 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_1 = paddle._C_ops.matmul(slice_1, transpose_2, False, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_1, full_3, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_2 = paddle._C_ops.matmul(softmax__0, slice_3, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_3 = paddle._C_ops.matmul(reshape__2, parameter_8, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__3 = paddle._C_ops.add_(matmul_3, parameter_9)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__4 = paddle._C_ops.add_(add__1, add__3)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__4, parameter_10, parameter_11, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_12, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__5 = paddle._C_ops.add_(matmul_4, parameter_13)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_0 = paddle._C_ops.gelu(add__5, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_14, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_15)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__7 = paddle._C_ops.add_(add__4, add__6)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__7, parameter_16, parameter_17, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_6 = paddle._C_ops.matmul(layer_norm_6, parameter_18, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_19)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_11 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__4, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_4, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_4, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_4, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_5 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_7 = paddle._C_ops.matmul(slice_4, transpose_5, False, False)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_7, full_4, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_8 = paddle._C_ops.matmul(softmax__1, slice_6, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_18 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_9 = paddle._C_ops.matmul(reshape__6, parameter_20, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__9 = paddle._C_ops.add_(matmul_9, parameter_21)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__10 = paddle._C_ops.add_(add__7, add__9)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__10, parameter_22, parameter_23, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_24, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__11 = paddle._C_ops.add_(matmul_10, parameter_25)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_1 = paddle._C_ops.gelu(add__11, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_26, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__12 = paddle._C_ops.add_(matmul_11, parameter_27)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__13 = paddle._C_ops.add_(add__10, add__12)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__13, parameter_28, parameter_29, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_12 = paddle._C_ops.matmul(layer_norm_12, parameter_30, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__14 = paddle._C_ops.add_(matmul_12, parameter_31)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_19 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__8, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_7, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_7, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_7, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_8 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_13 = paddle._C_ops.matmul(slice_7, transpose_8, False, False)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_13, full_5, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_14 = paddle._C_ops.matmul(softmax__2, slice_9, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_9 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_26 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_15 = paddle._C_ops.matmul(reshape__10, parameter_32, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__15 = paddle._C_ops.add_(matmul_15, parameter_33)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__16 = paddle._C_ops.add_(add__13, add__15)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__16, parameter_34, parameter_35, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_16 = paddle._C_ops.matmul(layer_norm_15, parameter_36, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__17 = paddle._C_ops.add_(matmul_16, parameter_37)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_2 = paddle._C_ops.gelu(add__17, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_38, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__18 = paddle._C_ops.add_(matmul_17, parameter_39)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__19 = paddle._C_ops.add_(add__16, add__18)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_40, parameter_41, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_18 = paddle._C_ops.matmul(layer_norm_18, parameter_42, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__20 = paddle._C_ops.add_(matmul_18, parameter_43)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_27 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__12, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_10, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_10, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_10, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_11 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_19 = paddle._C_ops.matmul(slice_10, transpose_11, False, False)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_19, full_6, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_20 = paddle._C_ops.matmul(softmax__3, slice_12, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_12 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_34 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_21 = paddle._C_ops.matmul(reshape__14, parameter_44, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__21 = paddle._C_ops.add_(matmul_21, parameter_45)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__22 = paddle._C_ops.add_(add__19, add__21)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_46, parameter_47, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_22 = paddle._C_ops.matmul(layer_norm_21, parameter_48, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__23 = paddle._C_ops.add_(matmul_22, parameter_49)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_3 = paddle._C_ops.gelu(add__23, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_50, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__24 = paddle._C_ops.add_(matmul_23, parameter_51)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__25 = paddle._C_ops.add_(add__22, add__24)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__25, parameter_52, parameter_53, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_24 = paddle._C_ops.matmul(layer_norm_24, parameter_54, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__26 = paddle._C_ops.add_(matmul_24, parameter_55)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_35 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_13 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_13, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_13, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_13, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_14 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_25 = paddle._C_ops.matmul(slice_13, transpose_14, False, False)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_25, full_7, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_26 = paddle._C_ops.matmul(softmax__4, slice_15, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_15 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_42 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_27 = paddle._C_ops.matmul(reshape__18, parameter_56, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__27 = paddle._C_ops.add_(matmul_27, parameter_57)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__28 = paddle._C_ops.add_(add__25, add__27)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_58, parameter_59, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_28 = paddle._C_ops.matmul(layer_norm_27, parameter_60, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__29 = paddle._C_ops.add_(matmul_28, parameter_61)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_4 = paddle._C_ops.gelu(add__29, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_62, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__30 = paddle._C_ops.add_(matmul_29, parameter_63)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__31 = paddle._C_ops.add_(add__28, add__30)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__31, parameter_64, parameter_65, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_30 = paddle._C_ops.matmul(layer_norm_30, parameter_66, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__32 = paddle._C_ops.add_(matmul_30, parameter_67)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_43 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__32, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_16 = paddle._C_ops.transpose(reshape__20, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_16, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_16, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_16, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_17 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_31 = paddle._C_ops.matmul(slice_16, transpose_17, False, False)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_31, full_8, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_32 = paddle._C_ops.matmul(softmax__5, slice_18, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_18 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_50 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_18, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_33 = paddle._C_ops.matmul(reshape__22, parameter_68, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__33 = paddle._C_ops.add_(matmul_33, parameter_69)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__34 = paddle._C_ops.add_(add__31, add__33)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__34, parameter_70, parameter_71, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_34 = paddle._C_ops.matmul(layer_norm_33, parameter_72, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__35 = paddle._C_ops.add_(matmul_34, parameter_73)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_5 = paddle._C_ops.gelu(add__35, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_74, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__36 = paddle._C_ops.add_(matmul_35, parameter_75)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__37 = paddle._C_ops.add_(add__34, add__36)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_76, parameter_77, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_36 = paddle._C_ops.matmul(layer_norm_36, parameter_78, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__38 = paddle._C_ops.add_(matmul_36, parameter_79)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_51 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_19 = paddle._C_ops.transpose(reshape__24, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_19, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_19, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_19, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_20 = paddle._C_ops.transpose(slice_20, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_37 = paddle._C_ops.matmul(slice_19, transpose_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_37, full_9, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_38 = paddle._C_ops.matmul(softmax__6, slice_21, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_21 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_58 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_39 = paddle._C_ops.matmul(reshape__26, parameter_80, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__39 = paddle._C_ops.add_(matmul_39, parameter_81)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__40 = paddle._C_ops.add_(add__37, add__39)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__40, parameter_82, parameter_83, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_40 = paddle._C_ops.matmul(layer_norm_39, parameter_84, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__41 = paddle._C_ops.add_(matmul_40, parameter_85)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_6 = paddle._C_ops.gelu(add__41, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_41 = paddle._C_ops.matmul(gelu_6, parameter_86, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__42 = paddle._C_ops.add_(matmul_41, parameter_87)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__43 = paddle._C_ops.add_(add__40, add__42)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_88, parameter_89, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_42 = paddle._C_ops.matmul(layer_norm_42, parameter_90, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__44 = paddle._C_ops.add_(matmul_42, parameter_91)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_59 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_22, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_22, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_22, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_23 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_43 = paddle._C_ops.matmul(slice_22, transpose_23, False, False)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_43, full_10, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_44 = paddle._C_ops.matmul(softmax__7, slice_24, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_24 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_66 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_45 = paddle._C_ops.matmul(reshape__30, parameter_92, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__45 = paddle._C_ops.add_(matmul_45, parameter_93)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__46 = paddle._C_ops.add_(add__43, add__45)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__46, parameter_94, parameter_95, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_46 = paddle._C_ops.matmul(layer_norm_45, parameter_96, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__47 = paddle._C_ops.add_(matmul_46, parameter_97)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_7 = paddle._C_ops.gelu(add__47, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_47 = paddle._C_ops.matmul(gelu_7, parameter_98, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__48 = paddle._C_ops.add_(matmul_47, parameter_99)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__49 = paddle._C_ops.add_(add__46, add__48)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_100, parameter_101, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_48 = paddle._C_ops.matmul(layer_norm_48, parameter_102, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__50 = paddle._C_ops.add_(matmul_48, parameter_103)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_67 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__32, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_25, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_25, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_25, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_26 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_49 = paddle._C_ops.matmul(slice_25, transpose_26, False, False)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_49, full_11, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__8 = paddle._C_ops.softmax_(scale__8, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_50 = paddle._C_ops.matmul(softmax__8, slice_27, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_27 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_74 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_51 = paddle._C_ops.matmul(reshape__34, parameter_104, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__51 = paddle._C_ops.add_(matmul_51, parameter_105)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__52 = paddle._C_ops.add_(add__49, add__51)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_106, parameter_107, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_52 = paddle._C_ops.matmul(layer_norm_51, parameter_108, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__53 = paddle._C_ops.add_(matmul_52, parameter_109)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_8 = paddle._C_ops.gelu(add__53, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_53 = paddle._C_ops.matmul(gelu_8, parameter_110, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__54 = paddle._C_ops.add_(matmul_53, parameter_111)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__55 = paddle._C_ops.add_(add__52, add__54)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__55, parameter_112, parameter_113, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_54 = paddle._C_ops.matmul(layer_norm_54, parameter_114, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__56 = paddle._C_ops.add_(matmul_54, parameter_115)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_75 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__36, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_28, [0], full_int_array_76, full_int_array_77, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_28, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_28, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_29 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_55 = paddle._C_ops.matmul(slice_28, transpose_29, False, False)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_55, full_12, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__9 = paddle._C_ops.softmax_(scale__9, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_56 = paddle._C_ops.matmul(softmax__9, slice_30, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_30 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_82 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_57 = paddle._C_ops.matmul(reshape__38, parameter_116, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__57 = paddle._C_ops.add_(matmul_57, parameter_117)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__58 = paddle._C_ops.add_(add__55, add__57)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__58, parameter_118, parameter_119, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_58 = paddle._C_ops.matmul(layer_norm_57, parameter_120, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__59 = paddle._C_ops.add_(matmul_58, parameter_121)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_9 = paddle._C_ops.gelu(add__59, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_59 = paddle._C_ops.matmul(gelu_9, parameter_122, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__60 = paddle._C_ops.add_(matmul_59, parameter_123)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__61 = paddle._C_ops.add_(add__58, add__60)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__61, parameter_124, parameter_125, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_60 = paddle._C_ops.matmul(layer_norm_60, parameter_126, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__62 = paddle._C_ops.add_(matmul_60, parameter_127)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_83 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__62, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_31, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_31, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_31, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_32 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_61 = paddle._C_ops.matmul(slice_31, transpose_32, False, False)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_61, full_13, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__10 = paddle._C_ops.softmax_(scale__10, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_62 = paddle._C_ops.matmul(softmax__10, slice_33, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_33 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_90 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_63 = paddle._C_ops.matmul(reshape__42, parameter_128, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__63 = paddle._C_ops.add_(matmul_63, parameter_129)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__64 = paddle._C_ops.add_(add__61, add__63)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__64, parameter_130, parameter_131, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_64 = paddle._C_ops.matmul(layer_norm_63, parameter_132, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__65 = paddle._C_ops.add_(matmul_64, parameter_133)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_10 = paddle._C_ops.gelu(add__65, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_65 = paddle._C_ops.matmul(gelu_10, parameter_134, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__66 = paddle._C_ops.add_(matmul_65, parameter_135)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__67 = paddle._C_ops.add_(add__64, add__66)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__67, parameter_136, parameter_137, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x2304xf16) <- (-1x197x768xf16, 768x2304xf16)
        matmul_66 = paddle._C_ops.matmul(layer_norm_66, parameter_138, False, False)

        # pd_op.add_: (-1x197x2304xf16) <- (-1x197x2304xf16, 2304xf16)
        add__68 = paddle._C_ops.add_(matmul_66, parameter_139)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_91 = [-1, 197, 3, 12, 64]

        # pd_op.reshape_: (-1x197x3x12x64xf16, 0x-1x197x2304xf16) <- (-1x197x2304xf16, 5xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__68, full_int_array_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x12x197x64xf16) <- (-1x197x3x12x64xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__44, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_34, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [2]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_34, [0], full_int_array_94, full_int_array_95, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [3]

        # pd_op.slice: (-1x12x197x64xf16) <- (3x-1x12x197x64xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_34, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.transpose: (-1x12x64x197xf16) <- (-1x12x197x64xf16)
        transpose_35 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf16) <- (-1x12x197x64xf16, -1x12x64x197xf16)
        matmul_67 = paddle._C_ops.matmul(slice_34, transpose_35, False, False)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x197x197xf16) <- (-1x12x197x197xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_67, full_14, float('0'), True)

        # pd_op.softmax_: (-1x12x197x197xf16) <- (-1x12x197x197xf16)
        softmax__11 = paddle._C_ops.softmax_(scale__11, -1)

        # pd_op.matmul: (-1x12x197x64xf16) <- (-1x12x197x197xf16, -1x12x197x64xf16)
        matmul_68 = paddle._C_ops.matmul(softmax__11, slice_36, False, False)

        # pd_op.transpose: (-1x197x12x64xf16) <- (-1x12x197x64xf16)
        transpose_36 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_98 = [-1, 197, 768]

        # pd_op.reshape_: (-1x197x768xf16, 0x-1x197x12x64xf16) <- (-1x197x12x64xf16, 3xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x768xf16, 768x768xf16)
        matmul_69 = paddle._C_ops.matmul(reshape__46, parameter_140, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__69 = paddle._C_ops.add_(matmul_69, parameter_141)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__70 = paddle._C_ops.add_(add__67, add__69)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__70, parameter_142, parameter_143, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x3072xf16) <- (-1x197x768xf16, 768x3072xf16)
        matmul_70 = paddle._C_ops.matmul(layer_norm_69, parameter_144, False, False)

        # pd_op.add_: (-1x197x3072xf16) <- (-1x197x3072xf16, 3072xf16)
        add__71 = paddle._C_ops.add_(matmul_70, parameter_145)

        # pd_op.gelu: (-1x197x3072xf16) <- (-1x197x3072xf16)
        gelu_11 = paddle._C_ops.gelu(add__71, False)

        # pd_op.matmul: (-1x197x768xf16) <- (-1x197x3072xf16, 3072x768xf16)
        matmul_71 = paddle._C_ops.matmul(gelu_11, parameter_146, False, False)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, 768xf16)
        add__72 = paddle._C_ops.add_(matmul_71, parameter_147)

        # pd_op.add_: (-1x197x768xf16) <- (-1x197x768xf16, -1x197x768xf16)
        add__73 = paddle._C_ops.add_(add__70, add__72)

        # pd_op.layer_norm: (-1x197x768xf16, -197xf32, -197xf32) <- (-1x197x768xf16, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__73, parameter_148, parameter_149, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [1]

        # pd_op.slice: (-1x768xf16) <- (-1x197x768xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(layer_norm_72, [1], full_int_array_99, full_int_array_100, [1], [1])

        # pd_op.matmul: (-1x1000xf16) <- (-1x768xf16, 768x1000xf16)
        matmul_72 = paddle._C_ops.matmul(slice_37, parameter_150, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__74 = paddle._C_ops.add_(matmul_72, parameter_151)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__12 = paddle._C_ops.softmax_(add__74, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_3 = paddle._C_ops.cast(softmax__12, paddle.float32)
        return cast_3



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_26, parameter_27, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_55, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_62, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_89, parameter_88, parameter_90, parameter_91, parameter_92, parameter_93, parameter_95, parameter_94, parameter_96, parameter_97, parameter_98, parameter_99, parameter_101, parameter_100, parameter_102, parameter_103, parameter_104, parameter_105, parameter_107, parameter_106, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_125, parameter_124, parameter_126, parameter_127, parameter_128, parameter_129, parameter_131, parameter_130, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, feed_0):
        return self.builtin_module_596_0_0(parameter_0, parameter_1, parameter_2, parameter_3, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_26, parameter_27, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_55, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_62, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_89, parameter_88, parameter_90, parameter_91, parameter_92, parameter_93, parameter_95, parameter_94, parameter_96, parameter_97, parameter_98, parameter_99, parameter_101, parameter_100, parameter_102, parameter_103, parameter_104, parameter_105, parameter_107, parameter_106, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_125, parameter_124, parameter_126, parameter_127, parameter_128, parameter_129, parameter_131, parameter_130, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_596_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([768, 3, 16, 16], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([768, 2304], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([768, 3072], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([3072, 768], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([768, 1000], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[768, 3, 16, 16], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_2
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[1, 197, 768], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[768, 2304], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[768, 3072], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[3072, 768], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[768, 1000], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32'),
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