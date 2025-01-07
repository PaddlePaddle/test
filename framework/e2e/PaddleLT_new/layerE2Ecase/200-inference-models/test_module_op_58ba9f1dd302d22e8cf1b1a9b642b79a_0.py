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
    return [676][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_861_0_0(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_47, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_55, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0):

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

        # pd_op.conv2d: (-1x32x56x56xf16) <- (-1x3x224x224xf16, 32x3x7x7xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf16) <- (-1x32x56x56xf16, 1x32x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.flatten_: (-1x32x3136xf16, None) <- (-1x32x56x56xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x32xf16) <- (-1x32x3136xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_0, parameter_4, parameter_5, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x32xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(layer_norm_3, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x32xf16, 32x32xf16)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_6, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__1 = paddle._C_ops.add_(matmul_0, parameter_7)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, full_0, full_1, full_2]

        # pd_op.reshape_: (-1x3136x1x32xf16, 0x-1x3136x32xf16) <- (-1x3136x32xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x32xf16) <- (-1x3136x1x32xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.transpose: (-1x32x3136xf16) <- (-1x3136x32xf16)
        transpose_2 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_3, full_4, full_5]

        # pd_op.reshape_: (-1x32x56x56xf16, 0x-1x32x3136xf16) <- (-1x32x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x32x7x7xf16) <- (-1x32x56x56xf16, 32x32x8x8xf16)
        conv2d_1 = paddle._C_ops.conv2d(reshape__2, parameter_8, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x7x7xf16) <- (-1x32x7x7xf16, 1x32x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, full_6, full_7]

        # pd_op.reshape_: (-1x32x49xf16, 0x-1x32x7x7xf16) <- (-1x32x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x32xf16) <- (-1x32x49xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x32xf16, -49xf32, -49xf32) <- (-1x49x32xf16, 32xf32, 32xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_3, parameter_10, parameter_11, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x64xf16) <- (-1x49x32xf16, 32x64xf16)
        matmul_1 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add_: (-1x49x64xf16) <- (-1x49x64xf16, 64xf16)
        add__3 = paddle._C_ops.add_(matmul_1, parameter_13)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, full_8, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x49x2x1x32xf16, 0x-1x49x64xf16) <- (-1x49x64xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__3, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x32xf16) <- (-1x49x2x1x32xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.slice: (-1x1x49x32xf16) <- (2x-1x1x49x32xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_4, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.slice: (-1x1x49x32xf16) <- (2x-1x1x49x32xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_4, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.transpose: (-1x1x32x49xf16) <- (-1x1x49x32xf16)
        transpose_5 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf16) <- (-1x1x3136x32xf16, -1x1x32x49xf16)
        matmul_2 = paddle._C_ops.matmul(transpose_1, transpose_5, False, False)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_2, full_12, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x1x3136x32xf16) <- (-1x1x3136x49xf16, -1x1x49x32xf16)
        matmul_3 = paddle._C_ops.matmul(softmax__0, slice_3, False, False)

        # pd_op.transpose: (-1x3136x1x32xf16) <- (-1x1x3136x32xf16)
        transpose_6 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, full_13, full_14]

        # pd_op.reshape_: (-1x3136x32xf16, 0x-1x3136x1x32xf16) <- (-1x3136x1x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x32xf16, 32x32xf16)
        matmul_4 = paddle._C_ops.matmul(reshape__8, parameter_14, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__4 = paddle._C_ops.add_(matmul_4, parameter_15)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, -1x3136x32xf16)
        add__5 = paddle._C_ops.add_(layer_norm_0, add__4)

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__5, parameter_16, parameter_17, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x256xf16) <- (-1x3136x32xf16, 32x256xf16)
        matmul_5 = paddle._C_ops.matmul(layer_norm_9, parameter_18, False, False)

        # pd_op.add_: (-1x3136x256xf16) <- (-1x3136x256xf16, 256xf16)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_19)

        # pd_op.shape: (3xi32) <- (-1x3136x256xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__6, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.transpose: (-1x256x3136xf16) <- (-1x3136x256xf16)
        transpose_7 = paddle._C_ops.transpose(add__6, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_4, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x256x56x56xf16, 0x-1x256x3136xf16) <- (-1x256x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 256x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(reshape__10, parameter_20, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__7 = paddle._C_ops.add_(depthwise_conv2d_0, reshape_4)

        # pd_op.flatten_: (-1x256x3136xf16, None) <- (-1x256x56x56xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__7, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x256xf16) <- (-1x256x3136xf16)
        transpose_8 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.gelu: (-1x3136x256xf16) <- (-1x3136x256xf16)
        gelu_0 = paddle._C_ops.gelu(transpose_8, False)

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x256xf16, 256x32xf16)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_22, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_23)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, -1x3136x32xf16)
        add__9 = paddle._C_ops.add_(add__5, add__8)

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__9, parameter_24, parameter_25, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x32xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(layer_norm_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_3, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x32xf16, 32x32xf16)
        matmul_7 = paddle._C_ops.matmul(layer_norm_12, parameter_26, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__10 = paddle._C_ops.add_(matmul_7, parameter_27)

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_5, full_18, full_19, full_20]

        # pd_op.reshape_: (-1x3136x1x32xf16, 0x-1x3136x32xf16) <- (-1x3136x32xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x32xf16) <- (-1x3136x1x32xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x32x3136xf16) <- (-1x3136x32xf16)
        transpose_10 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, full_21, full_22, full_23]

        # pd_op.reshape_: (-1x32x56x56xf16, 0x-1x32x3136xf16) <- (-1x32x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x32x7x7xf16) <- (-1x32x56x56xf16, 32x32x8x8xf16)
        conv2d_2 = paddle._C_ops.conv2d(reshape__14, parameter_28, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x7x7xf16) <- (-1x32x7x7xf16, 1x32x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_2, reshape_6)

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_5, full_24, full_25]

        # pd_op.reshape_: (-1x32x49xf16, 0x-1x32x7x7xf16) <- (-1x32x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x32xf16) <- (-1x32x49xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__16, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x32xf16, -49xf32, -49xf32) <- (-1x49x32xf16, 32xf32, 32xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_11, parameter_30, parameter_31, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x64xf16) <- (-1x49x32xf16, 32x64xf16)
        matmul_8 = paddle._C_ops.matmul(layer_norm_15, parameter_32, False, False)

        # pd_op.add_: (-1x49x64xf16) <- (-1x49x64xf16, 64xf16)
        add__12 = paddle._C_ops.add_(matmul_8, parameter_33)

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_5, full_26, full_27, full_28, full_29]

        # pd_op.reshape_: (-1x49x2x1x32xf16, 0x-1x49x64xf16) <- (-1x49x64xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__12, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x32xf16) <- (-1x49x2x1x32xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (-1x1x49x32xf16) <- (2x-1x1x49x32xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_12, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.slice: (-1x1x49x32xf16) <- (2x-1x1x49x32xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_12, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.transpose: (-1x1x32x49xf16) <- (-1x1x49x32xf16)
        transpose_13 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf16) <- (-1x1x3136x32xf16, -1x1x32x49xf16)
        matmul_9 = paddle._C_ops.matmul(transpose_9, transpose_13, False, False)

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_9, full_30, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x1x3136x32xf16) <- (-1x1x3136x49xf16, -1x1x49x32xf16)
        matmul_10 = paddle._C_ops.matmul(softmax__1, slice_7, False, False)

        # pd_op.transpose: (-1x3136x1x32xf16) <- (-1x1x3136x32xf16)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_5, full_31, full_32]

        # pd_op.reshape_: (-1x3136x32xf16, 0x-1x3136x1x32xf16) <- (-1x3136x1x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x32xf16, 32x32xf16)
        matmul_11 = paddle._C_ops.matmul(reshape__20, parameter_34, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__13 = paddle._C_ops.add_(matmul_11, parameter_35)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, -1x3136x32xf16)
        add__14 = paddle._C_ops.add_(add__9, add__13)

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__14, parameter_36, parameter_37, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x256xf16) <- (-1x3136x32xf16, 32x256xf16)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_38, False, False)

        # pd_op.add_: (-1x3136x256xf16) <- (-1x3136x256xf16, 256xf16)
        add__15 = paddle._C_ops.add_(matmul_12, parameter_39)

        # pd_op.shape: (3xi32) <- (-1x3136x256xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(add__15, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_4, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.transpose: (-1x256x3136xf16) <- (-1x3136x256xf16)
        transpose_15 = paddle._C_ops.transpose(add__15, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_8, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x256x56x56xf16, 0x-1x256x3136xf16) <- (-1x256x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 256x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(reshape__22, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, 1x256x1x1xf16)
        add__16 = paddle._C_ops.add_(depthwise_conv2d_1, reshape_8)

        # pd_op.flatten_: (-1x256x3136xf16, None) <- (-1x256x56x56xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__16, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x256xf16) <- (-1x256x3136xf16)
        transpose_16 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.gelu: (-1x3136x256xf16) <- (-1x3136x256xf16)
        gelu_1 = paddle._C_ops.gelu(transpose_16, False)

        # pd_op.matmul: (-1x3136x32xf16) <- (-1x3136x256xf16, 256x32xf16)
        matmul_13 = paddle._C_ops.matmul(gelu_1, parameter_42, False, False)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, 32xf16)
        add__17 = paddle._C_ops.add_(matmul_13, parameter_43)

        # pd_op.add_: (-1x3136x32xf16) <- (-1x3136x32xf16, -1x3136x32xf16)
        add__18 = paddle._C_ops.add_(add__14, add__17)

        # pd_op.layer_norm: (-1x3136x32xf16, -3136xf32, -3136xf32) <- (-1x3136x32xf16, 32xf32, 32xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__18, parameter_44, parameter_45, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_0, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x56x56x32xf16, 0x-1x3136x32xf16) <- (-1x3136x32xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_21, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x32x56x56xf16) <- (-1x56x56x32xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__24, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x32x56x56xf16, 64x32x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(transpose_17, parameter_46, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_47, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 1x64x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_3, reshape_10)

        # pd_op.flatten_: (-1x64x784xf16, None) <- (-1x64x28x28xf16)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__19, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x64xf16) <- (-1x64x784xf16)
        transpose_18 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_18, parameter_48, parameter_49, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_24, parameter_50, parameter_51, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x64xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(layer_norm_27, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_5, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x64xf16, 64x64xf16)
        matmul_14 = paddle._C_ops.matmul(layer_norm_27, parameter_52, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__20 = paddle._C_ops.add_(matmul_14, parameter_53)

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_9, full_39, full_40, full_41]

        # pd_op.reshape_: (-1x784x2x32xf16, 0x-1x784x64xf16) <- (-1x784x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x32xf16) <- (-1x784x2x32xf16)
        transpose_19 = paddle._C_ops.transpose(reshape__26, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x784xf16) <- (-1x784x64xf16)
        transpose_20 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_9, full_42, full_43, full_44]

        # pd_op.reshape_: (-1x64x28x28xf16, 0x-1x64x784xf16) <- (-1x64x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf16) <- (-1x64x28x28xf16, 64x64x4x4xf16)
        conv2d_4 = paddle._C_ops.conv2d(reshape__28, parameter_54, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x7x7xf16) <- (-1x64x7x7xf16, 1x64x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_4, reshape_12)

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_9, full_45, full_46]

        # pd_op.reshape_: (-1x64x49xf16, 0x-1x64x7x7xf16) <- (-1x64x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__21, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf16) <- (-1x64x49xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__30, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf16, -49xf32, -49xf32) <- (-1x49x64xf16, 64xf32, 64xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_21, parameter_56, parameter_57, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf16) <- (-1x49x64xf16, 64x128xf16)
        matmul_15 = paddle._C_ops.matmul(layer_norm_30, parameter_58, False, False)

        # pd_op.add_: (-1x49x128xf16) <- (-1x49x128xf16, 128xf16)
        add__22 = paddle._C_ops.add_(matmul_15, parameter_59)

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_9, full_47, full_48, full_49, full_50]

        # pd_op.reshape_: (-1x49x2x2x32xf16, 0x-1x49x128xf16) <- (-1x49x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x32xf16) <- (-1x49x2x2x32xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__32, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [1]

        # pd_op.slice: (-1x2x49x32xf16) <- (2x-1x2x49x32xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_22, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (-1x2x49x32xf16) <- (2x-1x2x49x32xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_22, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.transpose: (-1x2x32x49xf16) <- (-1x2x49x32xf16)
        transpose_23 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x32xf16, -1x2x32x49xf16)
        matmul_16 = paddle._C_ops.matmul(transpose_19, transpose_23, False, False)

        # pd_op.full: (1xf32) <- ()
        full_51 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_16, full_51, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x2x784x32xf16) <- (-1x2x784x49xf16, -1x2x49x32xf16)
        matmul_17 = paddle._C_ops.matmul(softmax__2, slice_11, False, False)

        # pd_op.transpose: (-1x784x2x32xf16) <- (-1x2x784x32xf16)
        transpose_24 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_9, full_52, full_53]

        # pd_op.reshape_: (-1x784x64xf16, 0x-1x784x2x32xf16) <- (-1x784x2x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x64xf16, 64x64xf16)
        matmul_18 = paddle._C_ops.matmul(reshape__34, parameter_60, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__23 = paddle._C_ops.add_(matmul_18, parameter_61)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, -1x784x64xf16)
        add__24 = paddle._C_ops.add_(layer_norm_24, add__23)

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__24, parameter_62, parameter_63, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x512xf16) <- (-1x784x64xf16, 64x512xf16)
        matmul_19 = paddle._C_ops.matmul(layer_norm_33, parameter_64, False, False)

        # pd_op.add_: (-1x784x512xf16) <- (-1x784x512xf16, 512xf16)
        add__25 = paddle._C_ops.add_(matmul_19, parameter_65)

        # pd_op.shape: (3xi32) <- (-1x784x512xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(add__25, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_6, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.transpose: (-1x512x784xf16) <- (-1x784x512xf16)
        transpose_25 = paddle._C_ops.transpose(add__25, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_12, full_54, full_55, full_56]

        # pd_op.reshape_: (-1x512x28x28xf16, 0x-1x512x784xf16) <- (-1x512x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 512x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(reshape__36, parameter_66, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_67, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__26 = paddle._C_ops.add_(depthwise_conv2d_2, reshape_14)

        # pd_op.flatten_: (-1x512x784xf16, None) <- (-1x512x28x28xf16)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__26, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x512xf16) <- (-1x512x784xf16)
        transpose_26 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.gelu: (-1x784x512xf16) <- (-1x784x512xf16)
        gelu_2 = paddle._C_ops.gelu(transpose_26, False)

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x512xf16, 512x64xf16)
        matmul_20 = paddle._C_ops.matmul(gelu_2, parameter_68, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__27 = paddle._C_ops.add_(matmul_20, parameter_69)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, -1x784x64xf16)
        add__28 = paddle._C_ops.add_(add__24, add__27)

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_70, parameter_71, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x64xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(layer_norm_36, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x64xf16, 64x64xf16)
        matmul_21 = paddle._C_ops.matmul(layer_norm_36, parameter_72, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__29 = paddle._C_ops.add_(matmul_21, parameter_73)

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_13, full_57, full_58, full_59]

        # pd_op.reshape_: (-1x784x2x32xf16, 0x-1x784x64xf16) <- (-1x784x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__29, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x32xf16) <- (-1x784x2x32xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__38, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x784xf16) <- (-1x784x64xf16)
        transpose_28 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_13, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x64x28x28xf16, 0x-1x64x784xf16) <- (-1x64x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf16) <- (-1x64x28x28xf16, 64x64x4x4xf16)
        conv2d_5 = paddle._C_ops.conv2d(reshape__40, parameter_74, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_75, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x7x7xf16) <- (-1x64x7x7xf16, 1x64x1x1xf16)
        add__30 = paddle._C_ops.add_(conv2d_5, reshape_16)

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_13, full_63, full_64]

        # pd_op.reshape_: (-1x64x49xf16, 0x-1x64x7x7xf16) <- (-1x64x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf16) <- (-1x64x49xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__42, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf16, -49xf32, -49xf32) <- (-1x49x64xf16, 64xf32, 64xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_29, parameter_76, parameter_77, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf16) <- (-1x49x64xf16, 64x128xf16)
        matmul_22 = paddle._C_ops.matmul(layer_norm_39, parameter_78, False, False)

        # pd_op.add_: (-1x49x128xf16) <- (-1x49x128xf16, 128xf16)
        add__31 = paddle._C_ops.add_(matmul_22, parameter_79)

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_13, full_65, full_66, full_67, full_68]

        # pd_op.reshape_: (-1x49x2x2x32xf16, 0x-1x49x128xf16) <- (-1x49x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__31, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x32xf16) <- (-1x49x2x2x32xf16)
        transpose_30 = paddle._C_ops.transpose(reshape__44, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.slice: (-1x2x49x32xf16) <- (2x-1x2x49x32xf16, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_30, [0], full_int_array_37, full_int_array_38, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2]

        # pd_op.slice: (-1x2x49x32xf16) <- (2x-1x2x49x32xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_30, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.transpose: (-1x2x32x49xf16) <- (-1x2x49x32xf16)
        transpose_31 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x32xf16, -1x2x32x49xf16)
        matmul_23 = paddle._C_ops.matmul(transpose_27, transpose_31, False, False)

        # pd_op.full: (1xf32) <- ()
        full_69 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_23, full_69, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x2x784x32xf16) <- (-1x2x784x49xf16, -1x2x49x32xf16)
        matmul_24 = paddle._C_ops.matmul(softmax__3, slice_15, False, False)

        # pd_op.transpose: (-1x784x2x32xf16) <- (-1x2x784x32xf16)
        transpose_32 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_23 = [slice_13, full_70, full_71]

        # pd_op.reshape_: (-1x784x64xf16, 0x-1x784x2x32xf16) <- (-1x784x2x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_32, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x64xf16, 64x64xf16)
        matmul_25 = paddle._C_ops.matmul(reshape__46, parameter_80, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__32 = paddle._C_ops.add_(matmul_25, parameter_81)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, -1x784x64xf16)
        add__33 = paddle._C_ops.add_(add__28, add__32)

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_82, parameter_83, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x512xf16) <- (-1x784x64xf16, 64x512xf16)
        matmul_26 = paddle._C_ops.matmul(layer_norm_42, parameter_84, False, False)

        # pd_op.add_: (-1x784x512xf16) <- (-1x784x512xf16, 512xf16)
        add__34 = paddle._C_ops.add_(matmul_26, parameter_85)

        # pd_op.shape: (3xi32) <- (-1x784x512xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add__34, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_8, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.transpose: (-1x512x784xf16) <- (-1x784x512xf16)
        transpose_33 = paddle._C_ops.transpose(add__34, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_16, full_72, full_73, full_74]

        # pd_op.reshape_: (-1x512x28x28xf16, 0x-1x512x784xf16) <- (-1x512x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 512x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__48, parameter_86, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_87, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__35 = paddle._C_ops.add_(depthwise_conv2d_3, reshape_18)

        # pd_op.flatten_: (-1x512x784xf16, None) <- (-1x512x28x28xf16)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__35, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x512xf16) <- (-1x512x784xf16)
        transpose_34 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.gelu: (-1x784x512xf16) <- (-1x784x512xf16)
        gelu_3 = paddle._C_ops.gelu(transpose_34, False)

        # pd_op.matmul: (-1x784x64xf16) <- (-1x784x512xf16, 512x64xf16)
        matmul_27 = paddle._C_ops.matmul(gelu_3, parameter_88, False, False)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, 64xf16)
        add__36 = paddle._C_ops.add_(matmul_27, parameter_89)

        # pd_op.add_: (-1x784x64xf16) <- (-1x784x64xf16, -1x784x64xf16)
        add__37 = paddle._C_ops.add_(add__33, add__36)

        # pd_op.layer_norm: (-1x784x64xf16, -784xf32, -784xf32) <- (-1x784x64xf16, 64xf32, 64xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_90, parameter_91, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_0, full_75, full_76, full_77]

        # pd_op.reshape_: (-1x28x28x64xf16, 0x-1x784x64xf16) <- (-1x784x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_45, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x28x28xf16) <- (-1x28x28x64xf16)
        transpose_35 = paddle._C_ops.transpose(reshape__50, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x160x14x14xf16) <- (-1x64x28x28xf16, 160x64x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(transpose_35, parameter_92, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf16, 0x160xf16) <- (160xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_93, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf16) <- (-1x160x14x14xf16, 1x160x1x1xf16)
        add__38 = paddle._C_ops.add_(conv2d_6, reshape_20)

        # pd_op.flatten_: (-1x160x196xf16, None) <- (-1x160x14x14xf16)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__38, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x160xf16) <- (-1x160x196xf16)
        transpose_36 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_36, parameter_94, parameter_95, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_48, parameter_96, parameter_97, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x160xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(layer_norm_51, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_9, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x160xf16, 160x160xf16)
        matmul_28 = paddle._C_ops.matmul(layer_norm_51, parameter_98, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__39 = paddle._C_ops.add_(matmul_28, parameter_99)

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_17, full_78, full_79, full_80]

        # pd_op.reshape_: (-1x196x5x32xf16, 0x-1x196x160xf16) <- (-1x196x160xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x32xf16) <- (-1x196x5x32xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3])

        # pd_op.transpose: (-1x160x196xf16) <- (-1x196x160xf16)
        transpose_38 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_17, full_81, full_82, full_83]

        # pd_op.reshape_: (-1x160x14x14xf16, 0x-1x160x196xf16) <- (-1x160x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, [x.reshape([]) for x in combine_27]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x7x7xf16) <- (-1x160x14x14xf16, 160x160x2x2xf16)
        conv2d_7 = paddle._C_ops.conv2d(reshape__54, parameter_100, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf16, 0x160xf16) <- (160xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x7x7xf16) <- (-1x160x7x7xf16, 1x160x1x1xf16)
        add__40 = paddle._C_ops.add_(conv2d_7, reshape_22)

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_17, full_84, full_85]

        # pd_op.reshape_: (-1x160x49xf16, 0x-1x160x7x7xf16) <- (-1x160x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__40, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x160xf16) <- (-1x160x49xf16)
        transpose_39 = paddle._C_ops.transpose(reshape__56, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x160xf16, -49xf32, -49xf32) <- (-1x49x160xf16, 160xf32, 160xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_39, parameter_102, parameter_103, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x320xf16) <- (-1x49x160xf16, 160x320xf16)
        matmul_29 = paddle._C_ops.matmul(layer_norm_54, parameter_104, False, False)

        # pd_op.add_: (-1x49x320xf16) <- (-1x49x320xf16, 320xf16)
        add__41 = paddle._C_ops.add_(matmul_29, parameter_105)

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_17, full_86, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x49x2x5x32xf16, 0x-1x49x320xf16) <- (-1x49x320xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__41, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x32xf16) <- (-1x49x2x5x32xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__58, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (-1x5x49x32xf16) <- (2x-1x5x49x32xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_40, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [2]

        # pd_op.slice: (-1x5x49x32xf16) <- (2x-1x5x49x32xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_40, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.transpose: (-1x5x32x49xf16) <- (-1x5x49x32xf16)
        transpose_41 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x32xf16, -1x5x32x49xf16)
        matmul_30 = paddle._C_ops.matmul(transpose_37, transpose_41, False, False)

        # pd_op.full: (1xf32) <- ()
        full_90 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_30, full_90, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x5x196x32xf16) <- (-1x5x196x49xf16, -1x5x49x32xf16)
        matmul_31 = paddle._C_ops.matmul(softmax__4, slice_19, False, False)

        # pd_op.transpose: (-1x196x5x32xf16) <- (-1x5x196x32xf16)
        transpose_42 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_17, full_91, full_92]

        # pd_op.reshape_: (-1x196x160xf16, 0x-1x196x5x32xf16) <- (-1x196x5x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x160xf16, 160x160xf16)
        matmul_32 = paddle._C_ops.matmul(reshape__60, parameter_106, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__42 = paddle._C_ops.add_(matmul_32, parameter_107)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, -1x196x160xf16)
        add__43 = paddle._C_ops.add_(layer_norm_48, add__42)

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_108, parameter_109, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x640xf16) <- (-1x196x160xf16, 160x640xf16)
        matmul_33 = paddle._C_ops.matmul(layer_norm_57, parameter_110, False, False)

        # pd_op.add_: (-1x196x640xf16) <- (-1x196x640xf16, 640xf16)
        add__44 = paddle._C_ops.add_(matmul_33, parameter_111)

        # pd_op.shape: (3xi32) <- (-1x196x640xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(add__44, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_10, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.transpose: (-1x640x196xf16) <- (-1x196x640xf16)
        transpose_43 = paddle._C_ops.transpose(add__44, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('640'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_20, full_93, full_94, full_95]

        # pd_op.reshape_: (-1x640x14x14xf16, 0x-1x640x196xf16) <- (-1x640x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_43, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(reshape__62, parameter_112, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf16, 0x640xf16) <- (640xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__45 = paddle._C_ops.add_(depthwise_conv2d_4, reshape_24)

        # pd_op.flatten_: (-1x640x196xf16, None) <- (-1x640x14x14xf16)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__45, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x640xf16) <- (-1x640x196xf16)
        transpose_44 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.gelu: (-1x196x640xf16) <- (-1x196x640xf16)
        gelu_4 = paddle._C_ops.gelu(transpose_44, False)

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x640xf16, 640x160xf16)
        matmul_34 = paddle._C_ops.matmul(gelu_4, parameter_114, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__46 = paddle._C_ops.add_(matmul_34, parameter_115)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, -1x196x160xf16)
        add__47 = paddle._C_ops.add_(add__43, add__46)

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__47, parameter_116, parameter_117, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x160xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(layer_norm_60, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_11, [0], full_int_array_55, full_int_array_56, [1], [0])

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x160xf16, 160x160xf16)
        matmul_35 = paddle._C_ops.matmul(layer_norm_60, parameter_118, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__48 = paddle._C_ops.add_(matmul_35, parameter_119)

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_21, full_96, full_97, full_98]

        # pd_op.reshape_: (-1x196x5x32xf16, 0x-1x196x160xf16) <- (-1x196x160xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x32xf16) <- (-1x196x5x32xf16)
        transpose_45 = paddle._C_ops.transpose(reshape__64, [0, 2, 1, 3])

        # pd_op.transpose: (-1x160x196xf16) <- (-1x196x160xf16)
        transpose_46 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_21, full_99, full_100, full_101]

        # pd_op.reshape_: (-1x160x14x14xf16, 0x-1x160x196xf16) <- (-1x160x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_46, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x7x7xf16) <- (-1x160x14x14xf16, 160x160x2x2xf16)
        conv2d_8 = paddle._C_ops.conv2d(reshape__66, parameter_120, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf16, 0x160xf16) <- (160xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_121, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x7x7xf16) <- (-1x160x7x7xf16, 1x160x1x1xf16)
        add__49 = paddle._C_ops.add_(conv2d_8, reshape_26)

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_34 = [slice_21, full_102, full_103]

        # pd_op.reshape_: (-1x160x49xf16, 0x-1x160x7x7xf16) <- (-1x160x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__49, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x160xf16) <- (-1x160x49xf16)
        transpose_47 = paddle._C_ops.transpose(reshape__68, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x160xf16, -49xf32, -49xf32) <- (-1x49x160xf16, 160xf32, 160xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_47, parameter_122, parameter_123, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x320xf16) <- (-1x49x160xf16, 160x320xf16)
        matmul_36 = paddle._C_ops.matmul(layer_norm_63, parameter_124, False, False)

        # pd_op.add_: (-1x49x320xf16) <- (-1x49x320xf16, 320xf16)
        add__50 = paddle._C_ops.add_(matmul_36, parameter_125)

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_21, full_104, full_105, full_106, full_107]

        # pd_op.reshape_: (-1x49x2x5x32xf16, 0x-1x49x320xf16) <- (-1x49x320xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x32xf16) <- (-1x49x2x5x32xf16)
        transpose_48 = paddle._C_ops.transpose(reshape__70, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [1]

        # pd_op.slice: (-1x5x49x32xf16) <- (2x-1x5x49x32xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_48, [0], full_int_array_58, full_int_array_59, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [2]

        # pd_op.slice: (-1x5x49x32xf16) <- (2x-1x5x49x32xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_48, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.transpose: (-1x5x32x49xf16) <- (-1x5x49x32xf16)
        transpose_49 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x32xf16, -1x5x32x49xf16)
        matmul_37 = paddle._C_ops.matmul(transpose_45, transpose_49, False, False)

        # pd_op.full: (1xf32) <- ()
        full_108 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_37, full_108, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x5x196x32xf16) <- (-1x5x196x49xf16, -1x5x49x32xf16)
        matmul_38 = paddle._C_ops.matmul(softmax__5, slice_23, False, False)

        # pd_op.transpose: (-1x196x5x32xf16) <- (-1x5x196x32xf16)
        transpose_50 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_21, full_109, full_110]

        # pd_op.reshape_: (-1x196x160xf16, 0x-1x196x5x32xf16) <- (-1x196x5x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_50, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x160xf16, 160x160xf16)
        matmul_39 = paddle._C_ops.matmul(reshape__72, parameter_126, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__51 = paddle._C_ops.add_(matmul_39, parameter_127)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, -1x196x160xf16)
        add__52 = paddle._C_ops.add_(add__47, add__51)

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_128, parameter_129, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x640xf16) <- (-1x196x160xf16, 160x640xf16)
        matmul_40 = paddle._C_ops.matmul(layer_norm_66, parameter_130, False, False)

        # pd_op.add_: (-1x196x640xf16) <- (-1x196x640xf16, 640xf16)
        add__53 = paddle._C_ops.add_(matmul_40, parameter_131)

        # pd_op.shape: (3xi32) <- (-1x196x640xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(add__53, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_12, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.transpose: (-1x640x196xf16) <- (-1x196x640xf16)
        transpose_51 = paddle._C_ops.transpose(add__53, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('640'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_24, full_111, full_112, full_113]

        # pd_op.reshape_: (-1x640x14x14xf16, 0x-1x640x196xf16) <- (-1x640x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_51, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(reshape__74, parameter_132, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_64 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf16, 0x640xf16) <- (640xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_133, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 1x640x1x1xf16)
        add__54 = paddle._C_ops.add_(depthwise_conv2d_5, reshape_28)

        # pd_op.flatten_: (-1x640x196xf16, None) <- (-1x640x14x14xf16)
        flatten__16, flatten__17 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__54, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x640xf16) <- (-1x640x196xf16)
        transpose_52 = paddle._C_ops.transpose(flatten__16, [0, 2, 1])

        # pd_op.gelu: (-1x196x640xf16) <- (-1x196x640xf16)
        gelu_5 = paddle._C_ops.gelu(transpose_52, False)

        # pd_op.matmul: (-1x196x160xf16) <- (-1x196x640xf16, 640x160xf16)
        matmul_41 = paddle._C_ops.matmul(gelu_5, parameter_134, False, False)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, 160xf16)
        add__55 = paddle._C_ops.add_(matmul_41, parameter_135)

        # pd_op.add_: (-1x196x160xf16) <- (-1x196x160xf16, -1x196x160xf16)
        add__56 = paddle._C_ops.add_(add__52, add__55)

        # pd_op.layer_norm: (-1x196x160xf16, -196xf32, -196xf32) <- (-1x196x160xf16, 160xf32, 160xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__56, parameter_136, parameter_137, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_0, full_114, full_115, full_116]

        # pd_op.reshape_: (-1x14x14x160xf16, 0x-1x196x160xf16) <- (-1x196x160xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_69, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x14x14xf16) <- (-1x14x14x160xf16)
        transpose_53 = paddle._C_ops.transpose(reshape__76, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x160x14x14xf16, 256x160x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(transpose_53, parameter_138, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 1x256x1x1xf16)
        add__57 = paddle._C_ops.add_(conv2d_9, reshape_30)

        # pd_op.flatten_: (-1x256x49xf16, None) <- (-1x256x7x7xf16)
        flatten__18, flatten__19 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__57, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x256xf16) <- (-1x256x49xf16)
        transpose_54 = paddle._C_ops.transpose(flatten__18, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_54, parameter_140, parameter_141, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_72, parameter_142, parameter_143, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x256xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(layer_norm_75, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_13, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x256xf16, 256x256xf16)
        matmul_42 = paddle._C_ops.matmul(layer_norm_75, parameter_144, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__58 = paddle._C_ops.add_(matmul_42, parameter_145)

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_25, full_117, full_118, full_119]

        # pd_op.reshape_: (-1x49x8x32xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__58, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x32xf16) <- (-1x49x8x32xf16)
        transpose_55 = paddle._C_ops.transpose(reshape__78, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x256xf16, 256x512xf16)
        matmul_43 = paddle._C_ops.matmul(layer_norm_75, parameter_146, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__59 = paddle._C_ops.add_(matmul_43, parameter_147)

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_25, full_120, full_121, full_122, full_123]

        # pd_op.reshape_: (-1x49x2x8x32xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__59, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x32xf16) <- (-1x49x2x8x32xf16)
        transpose_56 = paddle._C_ops.transpose(reshape__80, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (-1x8x49x32xf16) <- (2x-1x8x49x32xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_56, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [2]

        # pd_op.slice: (-1x8x49x32xf16) <- (2x-1x8x49x32xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_56, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.transpose: (-1x8x32x49xf16) <- (-1x8x49x32xf16)
        transpose_57 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x32xf16, -1x8x32x49xf16)
        matmul_44 = paddle._C_ops.matmul(transpose_55, transpose_57, False, False)

        # pd_op.full: (1xf32) <- ()
        full_124 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_44, full_124, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x8x49x32xf16) <- (-1x8x49x49xf16, -1x8x49x32xf16)
        matmul_45 = paddle._C_ops.matmul(softmax__6, slice_27, False, False)

        # pd_op.transpose: (-1x49x8x32xf16) <- (-1x8x49x32xf16)
        transpose_58 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_41 = [slice_25, full_125, full_126]

        # pd_op.reshape_: (-1x49x256xf16, 0x-1x49x8x32xf16) <- (-1x49x8x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_58, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x256xf16, 256x256xf16)
        matmul_46 = paddle._C_ops.matmul(reshape__82, parameter_148, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__60 = paddle._C_ops.add_(matmul_46, parameter_149)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, -1x49x256xf16)
        add__61 = paddle._C_ops.add_(layer_norm_72, add__60)

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__61, parameter_150, parameter_151, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x1024xf16) <- (-1x49x256xf16, 256x1024xf16)
        matmul_47 = paddle._C_ops.matmul(layer_norm_78, parameter_152, False, False)

        # pd_op.add_: (-1x49x1024xf16) <- (-1x49x1024xf16, 1024xf16)
        add__62 = paddle._C_ops.add_(matmul_47, parameter_153)

        # pd_op.shape: (3xi32) <- (-1x49x1024xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(add__62, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_14, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.transpose: (-1x1024x49xf16) <- (-1x49x1024xf16)
        transpose_59 = paddle._C_ops.transpose(add__62, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_28, full_127, full_128, full_129]

        # pd_op.reshape_: (-1x1024x7x7xf16, 0x-1x1024x49xf16) <- (-1x1024x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_59, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(reshape__84, parameter_154, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_74 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_155, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__63 = paddle._C_ops.add_(depthwise_conv2d_6, reshape_32)

        # pd_op.flatten_: (-1x1024x49xf16, None) <- (-1x1024x7x7xf16)
        flatten__20, flatten__21 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__63, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x1024xf16) <- (-1x1024x49xf16)
        transpose_60 = paddle._C_ops.transpose(flatten__20, [0, 2, 1])

        # pd_op.gelu: (-1x49x1024xf16) <- (-1x49x1024xf16)
        gelu_6 = paddle._C_ops.gelu(transpose_60, False)

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x1024xf16, 1024x256xf16)
        matmul_48 = paddle._C_ops.matmul(gelu_6, parameter_156, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__64 = paddle._C_ops.add_(matmul_48, parameter_157)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, -1x49x256xf16)
        add__65 = paddle._C_ops.add_(add__61, add__64)

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__65, parameter_158, parameter_159, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x256xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(layer_norm_81, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_15, [0], full_int_array_75, full_int_array_76, [1], [0])

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x256xf16, 256x256xf16)
        matmul_49 = paddle._C_ops.matmul(layer_norm_81, parameter_160, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__66 = paddle._C_ops.add_(matmul_49, parameter_161)

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_29, full_130, full_131, full_132]

        # pd_op.reshape_: (-1x49x8x32xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__66, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x32xf16) <- (-1x49x8x32xf16)
        transpose_61 = paddle._C_ops.transpose(reshape__86, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x256xf16, 256x512xf16)
        matmul_50 = paddle._C_ops.matmul(layer_norm_81, parameter_162, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__67 = paddle._C_ops.add_(matmul_50, parameter_163)

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_29, full_133, full_134, full_135, full_136]

        # pd_op.reshape_: (-1x49x2x8x32xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__67, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x32xf16) <- (-1x49x2x8x32xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__88, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [1]

        # pd_op.slice: (-1x8x49x32xf16) <- (2x-1x8x49x32xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_62, [0], full_int_array_77, full_int_array_78, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [2]

        # pd_op.slice: (-1x8x49x32xf16) <- (2x-1x8x49x32xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_62, [0], full_int_array_79, full_int_array_80, [1], [0])

        # pd_op.transpose: (-1x8x32x49xf16) <- (-1x8x49x32xf16)
        transpose_63 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x32xf16, -1x8x32x49xf16)
        matmul_51 = paddle._C_ops.matmul(transpose_61, transpose_63, False, False)

        # pd_op.full: (1xf32) <- ()
        full_137 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_51, full_137, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x8x49x32xf16) <- (-1x8x49x49xf16, -1x8x49x32xf16)
        matmul_52 = paddle._C_ops.matmul(softmax__7, slice_31, False, False)

        # pd_op.transpose: (-1x49x8x32xf16) <- (-1x8x49x32xf16)
        transpose_64 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_45 = [slice_29, full_138, full_139]

        # pd_op.reshape_: (-1x49x256xf16, 0x-1x49x8x32xf16) <- (-1x49x8x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_64, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x256xf16, 256x256xf16)
        matmul_53 = paddle._C_ops.matmul(reshape__90, parameter_164, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__68 = paddle._C_ops.add_(matmul_53, parameter_165)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, -1x49x256xf16)
        add__69 = paddle._C_ops.add_(add__65, add__68)

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__69, parameter_166, parameter_167, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x1024xf16) <- (-1x49x256xf16, 256x1024xf16)
        matmul_54 = paddle._C_ops.matmul(layer_norm_84, parameter_168, False, False)

        # pd_op.add_: (-1x49x1024xf16) <- (-1x49x1024xf16, 1024xf16)
        add__70 = paddle._C_ops.add_(matmul_54, parameter_169)

        # pd_op.shape: (3xi32) <- (-1x49x1024xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(add__70, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_16, [0], full_int_array_81, full_int_array_82, [1], [0])

        # pd_op.transpose: (-1x1024x49xf16) <- (-1x49x1024xf16)
        transpose_65 = paddle._C_ops.transpose(add__70, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_32, full_140, full_141, full_142]

        # pd_op.reshape_: (-1x1024x7x7xf16, 0x-1x1024x49xf16) <- (-1x1024x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_65, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1x3x3xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(reshape__92, parameter_170, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_83 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_171, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__71 = paddle._C_ops.add_(depthwise_conv2d_7, reshape_34)

        # pd_op.flatten_: (-1x1024x49xf16, None) <- (-1x1024x7x7xf16)
        flatten__22, flatten__23 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__71, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x1024xf16) <- (-1x1024x49xf16)
        transpose_66 = paddle._C_ops.transpose(flatten__22, [0, 2, 1])

        # pd_op.gelu: (-1x49x1024xf16) <- (-1x49x1024xf16)
        gelu_7 = paddle._C_ops.gelu(transpose_66, False)

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x1024xf16, 1024x256xf16)
        matmul_55 = paddle._C_ops.matmul(gelu_7, parameter_172, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__72 = paddle._C_ops.add_(matmul_55, parameter_173)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, -1x49x256xf16)
        add__73 = paddle._C_ops.add_(add__69, add__72)

        # pd_op.layer_norm: (-1x49x256xf16, -49xf32, -49xf32) <- (-1x49x256xf16, 256xf32, 256xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__73, parameter_174, parameter_175, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x256xf16) <- (-1x49x256xf16)
        mean_0 = paddle._C_ops.mean(layer_norm_87, [1], False)

        # pd_op.matmul: (-1x1000xf16) <- (-1x256xf16, 256x1000xf16)
        matmul_56 = paddle._C_ops.matmul(mean_0, parameter_176, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__74 = paddle._C_ops.add_(matmul_56, parameter_177)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__8 = paddle._C_ops.softmax_(add__74, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__8, paddle.float32)
        return cast_1



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

    def forward(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_47, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_55, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0):
        return self.builtin_module_861_0_0(parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_45, parameter_44, parameter_46, parameter_47, parameter_49, parameter_48, parameter_51, parameter_50, parameter_52, parameter_53, parameter_54, parameter_55, parameter_57, parameter_56, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_95, parameter_94, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_123, parameter_122, parameter_124, parameter_125, parameter_126, parameter_127, parameter_129, parameter_128, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_861_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32, 32], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32, 32, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32, 64], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32, 32], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32, 256], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 32], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32, 32], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32, 32, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32, 64], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32, 32], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32, 256], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256, 32], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64, 64, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64, 128], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 512], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512, 64], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([64, 64, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64, 128], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64, 512], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512, 64], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([160, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([160, 160], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([160, 160, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([160, 320], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([160, 160], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([160, 640], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([640, 160], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([160, 160], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([160, 160, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([160, 320], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([160, 160], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([160, 640], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([640, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([640, 160], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([160], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256, 160, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256, 512], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256, 1024], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1024, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1024, 256], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 512], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256, 1024], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1024, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([1024, 256], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256, 1000], dtype='float16', min=0, max=0.5),
            # parameter_177
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
            paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32, 32], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[32, 32, 8, 8], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32, 64], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[32, 32], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32, 256], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 32], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32, 32], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[32, 32, 8, 8], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32, 64], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[32, 32], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32, 256], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[256, 32], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[64, 64, 4, 4], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64, 128], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 512], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[512, 64], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[64, 64, 4, 4], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[64, 128], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[64, 512], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[512, 64], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[160, 64, 3, 3], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[160, 160], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[160, 160, 2, 2], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[160, 320], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[160, 160], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[160, 640], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[640, 160], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[160, 160], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[160, 160, 2, 2], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[160, 320], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[160, 160], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[160, 640], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[640, 160], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[160], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256, 160, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[256, 512], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256, 1024], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[1024, 256], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 512], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256, 1024], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[1024, 256], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[256, 1000], dtype='float16'),
            # parameter_177
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