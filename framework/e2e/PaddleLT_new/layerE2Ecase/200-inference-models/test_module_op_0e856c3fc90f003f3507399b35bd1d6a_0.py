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
    return [426][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_629_0_0(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_7, parameter_4, parameter_6, parameter_5, parameter_8, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_163, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, feed_0):

        # pd_op.shape: (5xi32) <- (-1x2x350x25x1xf32)
        shape_0 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (5xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.transpose: (-1x1x2x350x25xf32) <- (-1x2x350x25x1xf32)
        transpose_0 = paddle._C_ops.transpose(feed_0, [0, 4, 1, 2, 3])

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float('0'), True)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [scale_0, full_1, full_2, full_3]

        # pd_op.reshape_: (-1x2x350x25xf32, 0x-1x1x2x350x25xf32) <- (-1x1x2x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x192x350x25xf32) <- (-1x2x350x25xf32, 192x2x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape__0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x350x25xf32) <- (-1x192x350x25xf32, 1x192x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.shape: (4xi32) <- (-1x192x350x25xf32)
        shape_1 = paddle._C_ops.shape(add__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_4, full_5, full_6, full_7]

        # pd_op.reshape_: (-1x64x3x350x25xf32, 0x-1x192x350x25xf32) <- (-1x192x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x3x25x350xf32) <- (-1x64x3x350x25xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, full_8, full_9, full_10]

        # pd_op.reshape_: (-1x64x75x350xf32, 0x-1x64x3x25x350xf32) <- (-1x64x3x25x350xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x64x350xf32) <- (-1x64x75x350xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x64x350xf32) <- (-1x75x64x350xf32, 25x75x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(transpose_2, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x64x350xf32) <- (-1x25x64x350xf32, 1x25x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.transpose: (-1x64x350x25xf32) <- (-1x25x64x350xf32)
        transpose_3 = paddle._C_ops.transpose(add__1, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_3, parameter_4, parameter_5, parameter_6, parameter_7, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__0, parameter_8, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_2, reshape_4)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__2, parameter_10, parameter_11, parameter_12, parameter_13, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__1, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_3, reshape_6)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x192x350x25xf32) <- (-1x64x350x25xf32, 192x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__1, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x350x25xf32) <- (-1x192x350x25xf32, 1x192x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_4, reshape_8)

        # pd_op.shape: (4xi32) <- (-1x192x350x25xf32)
        shape_2 = paddle._C_ops.shape(add__4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_2, full_11, full_12, full_13, full_14]

        # pd_op.reshape_: (-1x64x3x350x25xf32, 0x-1x192x350x25xf32) <- (-1x192x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__4, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x3x25x350xf32) <- (-1x64x3x350x25xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__6, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_2, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x64x75x350xf32, 0x-1x64x3x25x350xf32) <- (-1x64x3x25x350xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x64x350xf32) <- (-1x64x75x350xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x64x350xf32) <- (-1x75x64x350xf32, 25x75x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(transpose_5, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_23, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x64x350xf32) <- (-1x25x64x350xf32, 1x25x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_5, reshape_10)

        # pd_op.transpose: (-1x64x350x25xf32) <- (-1x25x64x350xf32)
        transpose_6 = paddle._C_ops.transpose(add__5, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_6, parameter_24, parameter_25, parameter_26, parameter_27, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__2, parameter_28, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_6, reshape_12)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, -1x64x350x25xf32)
        add__7 = paddle._C_ops.add_(batch_norm__24, batch_norm__12)

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__3 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__3, parameter_34, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_35, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_7, reshape_14)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__8, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x192x350x25xf32) <- (-1x64x350x25xf32, 192x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__3, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x350x25xf32) <- (-1x192x350x25xf32, 1x192x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_8, reshape_16)

        # pd_op.shape: (4xi32) <- (-1x192x350x25xf32)
        shape_3 = paddle._C_ops.shape(add__9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_3, full_18, full_19, full_20, full_21]

        # pd_op.reshape_: (-1x64x3x350x25xf32, 0x-1x192x350x25xf32) <- (-1x192x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__9, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x3x25x350xf32) <- (-1x64x3x350x25xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__10, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_3, full_22, full_23, full_24]

        # pd_op.reshape_: (-1x64x75x350xf32, 0x-1x64x3x25x350xf32) <- (-1x64x3x25x350xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x64x350xf32) <- (-1x64x75x350xf32)
        transpose_8 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x64x350xf32) <- (-1x75x64x350xf32, 25x75x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(transpose_8, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_43, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x64x350xf32) <- (-1x25x64x350xf32, 1x25x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_9, reshape_18)

        # pd_op.transpose: (-1x64x350x25xf32) <- (-1x25x64x350xf32)
        transpose_9 = paddle._C_ops.transpose(add__10, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_9, parameter_44, parameter_45, parameter_46, parameter_47, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__4, parameter_48, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_49, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_10, reshape_20)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__11, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, -1x64x350x25xf32)
        add__12 = paddle._C_ops.add_(batch_norm__42, batch_norm__30)

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__5 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__5, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_11, reshape_22)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__13, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x192x350x25xf32) <- (-1x64x350x25xf32, 192x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__5, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_61, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x350x25xf32) <- (-1x192x350x25xf32, 1x192x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_12, reshape_24)

        # pd_op.shape: (4xi32) <- (-1x192x350x25xf32)
        shape_4 = paddle._C_ops.shape(add__14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_4, full_25, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x64x3x350x25xf32, 0x-1x192x350x25xf32) <- (-1x192x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x3x25x350xf32) <- (-1x64x3x350x25xf32)
        transpose_10 = paddle._C_ops.transpose(reshape__14, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_4, full_29, full_30, full_31]

        # pd_op.reshape_: (-1x64x75x350xf32, 0x-1x64x3x25x350xf32) <- (-1x64x3x25x350xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x64x350xf32) <- (-1x64x75x350xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x64x350xf32) <- (-1x75x64x350xf32, 25x75x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(transpose_11, parameter_62, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_63, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x64x350xf32) <- (-1x25x64x350xf32, 1x25x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_13, reshape_26)

        # pd_op.transpose: (-1x64x350x25xf32) <- (-1x25x64x350xf32)
        transpose_12 = paddle._C_ops.transpose(add__15, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_12, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__6, parameter_68, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, 1x64x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_14, reshape_28)

        # pd_op.batch_norm_: (-1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x350x25xf32) <- (-1x64x350x25xf32, -1x64x350x25xf32)
        add__17 = paddle._C_ops.add_(batch_norm__60, batch_norm__48)

        # pd_op.relu_: (-1x64x350x25xf32) <- (-1x64x350x25xf32)
        relu__7 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__7, parameter_74, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_75, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_15, reshape_30)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__18, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x350x25xf32) <- (-1x64x350x25xf32, 384x64x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__7, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_81, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x350x25xf32) <- (-1x384x350x25xf32, 1x384x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_16, reshape_32)

        # pd_op.shape: (4xi32) <- (-1x384x350x25xf32)
        shape_5 = paddle._C_ops.shape(add__19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_5, full_32, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x128x3x350x25xf32, 0x-1x384x350x25xf32) <- (-1x384x350x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__19, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x3x25x350xf32) <- (-1x128x3x350x25xf32)
        transpose_13 = paddle._C_ops.transpose(reshape__18, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('350'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_5, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x128x75x350xf32, 0x-1x128x3x25x350xf32) <- (-1x128x3x25x350xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x128x350xf32) <- (-1x128x75x350xf32)
        transpose_14 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x128x350xf32) <- (-1x75x128x350xf32, 25x75x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(transpose_14, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_83, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x128x350xf32) <- (-1x25x128x350xf32, 1x25x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_17, reshape_34)

        # pd_op.transpose: (-1x128x350x25xf32) <- (-1x25x128x350xf32)
        transpose_15 = paddle._C_ops.transpose(add__20, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x128x350x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x350x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_15, parameter_84, parameter_85, parameter_86, parameter_87, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x350x25xf32) <- (-1x128x350x25xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x128x350x25xf32, 128x128x9x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__8, parameter_88, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_18, reshape_36)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__21, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, -1x128x175x25xf32)
        add__22 = paddle._C_ops.add_(batch_norm__78, batch_norm__66)

        # pd_op.relu_: (-1x128x175x25xf32) <- (-1x128x175x25xf32)
        relu__9 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__9, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_95, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_19, reshape_38)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x175x25xf32) <- (-1x128x175x25xf32, 384x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__9, parameter_100, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x175x25xf32) <- (-1x384x175x25xf32, 1x384x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_20, reshape_40)

        # pd_op.shape: (4xi32) <- (-1x384x175x25xf32)
        shape_6 = paddle._C_ops.shape(add__24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_6, full_39, full_40, full_41, full_42]

        # pd_op.reshape_: (-1x128x3x175x25xf32, 0x-1x384x175x25xf32) <- (-1x384x175x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__24, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x3x25x175xf32) <- (-1x128x3x175x25xf32)
        transpose_16 = paddle._C_ops.transpose(reshape__22, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_6, full_43, full_44, full_45]

        # pd_op.reshape_: (-1x128x75x175xf32, 0x-1x128x3x25x175xf32) <- (-1x128x3x25x175xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x128x175xf32) <- (-1x128x75x175xf32)
        transpose_17 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x128x175xf32) <- (-1x75x128x175xf32, 25x75x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(transpose_17, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x128x175xf32) <- (-1x25x128x175xf32, 1x25x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_21, reshape_42)

        # pd_op.transpose: (-1x128x175x25xf32) <- (-1x25x128x175xf32)
        transpose_18 = paddle._C_ops.transpose(add__25, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_18, parameter_104, parameter_105, parameter_106, parameter_107, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x175x25xf32) <- (-1x128x175x25xf32)
        relu__10 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 128x128x9x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__10, parameter_108, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_109, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_22, reshape_44)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__26, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, -1x128x175x25xf32)
        add__27 = paddle._C_ops.add_(batch_norm__96, batch_norm__84)

        # pd_op.relu_: (-1x128x175x25xf32) <- (-1x128x175x25xf32)
        relu__11 = paddle._C_ops.relu_(add__27)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__11, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_23, reshape_46)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__28, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x175x25xf32) <- (-1x128x175x25xf32, 384x128x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__11, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_121, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x175x25xf32) <- (-1x384x175x25xf32, 1x384x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_24, reshape_48)

        # pd_op.shape: (4xi32) <- (-1x384x175x25xf32)
        shape_7 = paddle._C_ops.shape(add__29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_7, full_46, full_47, full_48, full_49]

        # pd_op.reshape_: (-1x128x3x175x25xf32, 0x-1x384x175x25xf32) <- (-1x384x175x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__29, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x3x25x175xf32) <- (-1x128x3x175x25xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__26, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_7, full_50, full_51, full_52]

        # pd_op.reshape_: (-1x128x75x175xf32, 0x-1x128x3x25x175xf32) <- (-1x128x3x25x175xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x128x175xf32) <- (-1x128x75x175xf32)
        transpose_20 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x128x175xf32) <- (-1x75x128x175xf32, 25x75x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(transpose_20, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_123, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x128x175xf32) <- (-1x25x128x175xf32, 1x25x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_25, reshape_50)

        # pd_op.transpose: (-1x128x175x25xf32) <- (-1x25x128x175xf32)
        transpose_21 = paddle._C_ops.transpose(add__30, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_21, parameter_124, parameter_125, parameter_126, parameter_127, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x175x25xf32) <- (-1x128x175x25xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 128x128x9x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__12, parameter_128, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_129, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, 1x128x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_26, reshape_52)

        # pd_op.batch_norm_: (-1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__31, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x128x175x25xf32) <- (-1x128x175x25xf32, -1x128x175x25xf32)
        add__32 = paddle._C_ops.add_(batch_norm__114, batch_norm__102)

        # pd_op.relu_: (-1x128x175x25xf32) <- (-1x128x175x25xf32)
        relu__13 = paddle._C_ops.relu_(add__32)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__13, parameter_134, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_135, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_27, reshape_54)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__33, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x768x175x25xf32) <- (-1x128x175x25xf32, 768x128x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__13, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32, 0x768xf32) <- (768xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_141, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x175x25xf32) <- (-1x768x175x25xf32, 1x768x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_28, reshape_56)

        # pd_op.shape: (4xi32) <- (-1x768x175x25xf32)
        shape_8 = paddle._C_ops.shape(add__34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_8, full_53, full_54, full_55, full_56]

        # pd_op.reshape_: (-1x256x3x175x25xf32, 0x-1x768x175x25xf32) <- (-1x768x175x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__34, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x3x25x175xf32) <- (-1x256x3x175x25xf32)
        transpose_22 = paddle._C_ops.transpose(reshape__30, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('175'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_8, full_57, full_58, full_59]

        # pd_op.reshape_: (-1x256x75x175xf32, 0x-1x256x3x25x175xf32) <- (-1x256x3x25x175xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x256x175xf32) <- (-1x256x75x175xf32)
        transpose_23 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x256x175xf32) <- (-1x75x256x175xf32, 25x75x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(transpose_23, parameter_142, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_143, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x256x175xf32) <- (-1x25x256x175xf32, 1x25x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_29, reshape_58)

        # pd_op.transpose: (-1x256x175x25xf32) <- (-1x25x256x175xf32)
        transpose_24 = paddle._C_ops.transpose(add__35, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x256x175x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x175x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_24, parameter_144, parameter_145, parameter_146, parameter_147, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x175x25xf32) <- (-1x256x175x25xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x256x175x25xf32, 256x256x9x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__14, parameter_148, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_149, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_30, reshape_60)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__36, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, -1x256x88x25xf32)
        add__37 = paddle._C_ops.add_(batch_norm__132, batch_norm__120)

        # pd_op.relu_: (-1x256x88x25xf32) <- (-1x256x88x25xf32)
        relu__15 = paddle._C_ops.relu_(add__37)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__15, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_155, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_31, reshape_62)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__38, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x768x88x25xf32) <- (-1x256x88x25xf32, 768x256x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__15, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32, 0x768xf32) <- (768xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x88x25xf32) <- (-1x768x88x25xf32, 1x768x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_32, reshape_64)

        # pd_op.shape: (4xi32) <- (-1x768x88x25xf32)
        shape_9 = paddle._C_ops.shape(add__39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_9, [0], full_int_array_51, full_int_array_52, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('88'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_9, full_60, full_61, full_62, full_63]

        # pd_op.reshape_: (-1x256x3x88x25xf32, 0x-1x768x88x25xf32) <- (-1x768x88x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x3x25x88xf32) <- (-1x256x3x88x25xf32)
        transpose_25 = paddle._C_ops.transpose(reshape__34, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('88'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_9, full_64, full_65, full_66]

        # pd_op.reshape_: (-1x256x75x88xf32, 0x-1x256x3x25x88xf32) <- (-1x256x3x25x88xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x256x88xf32) <- (-1x256x75x88xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x256x88xf32) <- (-1x75x256x88xf32, 25x75x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(transpose_26, parameter_162, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_163, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x256x88xf32) <- (-1x25x256x88xf32, 1x25x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_33, reshape_66)

        # pd_op.transpose: (-1x256x88x25xf32) <- (-1x25x256x88xf32)
        transpose_27 = paddle._C_ops.transpose(add__40, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_27, parameter_164, parameter_165, parameter_166, parameter_167, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x88x25xf32) <- (-1x256x88x25xf32)
        relu__16 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 256x256x9x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__16, parameter_168, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_169, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__41 = paddle._C_ops.add_(conv2d_34, reshape_68)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__41, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, -1x256x88x25xf32)
        add__42 = paddle._C_ops.add_(batch_norm__150, batch_norm__138)

        # pd_op.relu_: (-1x256x88x25xf32) <- (-1x256x88x25xf32)
        relu__17 = paddle._C_ops.relu_(add__42)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__17, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_175, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_35, reshape_70)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__43, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x768x88x25xf32) <- (-1x256x88x25xf32, 768x256x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__17, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32, 0x768xf32) <- (768xf32, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x88x25xf32) <- (-1x768x88x25xf32, 1x768x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_36, reshape_72)

        # pd_op.shape: (4xi32) <- (-1x768x88x25xf32)
        shape_10 = paddle._C_ops.shape(add__44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_10, [0], full_int_array_57, full_int_array_58, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('88'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('25'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_10, full_67, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x256x3x88x25xf32, 0x-1x768x88x25xf32) <- (-1x768x88x25xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x256x3x25x88xf32) <- (-1x256x3x88x25xf32)
        transpose_28 = paddle._C_ops.transpose(reshape__38, [0, 1, 2, 4, 3])

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('75'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('88'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_10, full_71, full_72, full_73]

        # pd_op.reshape_: (-1x256x75x88xf32, 0x-1x256x3x25x88xf32) <- (-1x256x3x25x88xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x75x256x88xf32) <- (-1x256x75x88xf32)
        transpose_29 = paddle._C_ops.transpose(reshape__40, [0, 2, 1, 3])

        # pd_op.conv2d: (-1x25x256x88xf32) <- (-1x75x256x88xf32, 25x75x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(transpose_29, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 25, 1, 1]

        # pd_op.reshape: (1x25x1x1xf32, 0x25xf32) <- (25xf32, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x25x256x88xf32) <- (-1x25x256x88xf32, 1x25x1x1xf32)
        add__45 = paddle._C_ops.add_(conv2d_37, reshape_74)

        # pd_op.transpose: (-1x256x88x25xf32) <- (-1x25x256x88xf32)
        transpose_30 = paddle._C_ops.transpose(add__45, [0, 2, 3, 1])

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(transpose_30, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x88x25xf32) <- (-1x256x88x25xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 256x256x9x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__18, parameter_188, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, 1x256x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_38, reshape_76)

        # pd_op.batch_norm_: (-1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__46, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x88x25xf32) <- (-1x256x88x25xf32, -1x256x88x25xf32)
        add__47 = paddle._C_ops.add_(batch_norm__168, batch_norm__156)

        # pd_op.relu_: (-1x256x88x25xf32) <- (-1x256x88x25xf32)
        relu__19 = paddle._C_ops.relu_(add__47)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_61 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x88x25xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__19, full_int_array_61, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_0, full_74, full_75, full_76, full_77]

        # pd_op.reshape_: (-1x1x256x1x1xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_0, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.mean: (-1x256x1x1xf32) <- (-1x1x256x1x1xf32)
        mean_0 = paddle._C_ops.mean(reshape__42, [1], False)

        # pd_op.conv2d: (-1x30x1x1xf32) <- (-1x256x1x1xf32, 30x256x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(mean_0, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 30, 1, 1]

        # pd_op.reshape: (1x30x1x1xf32, 0x30xf32) <- (30xf32, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x30x1x1xf32) <- (-1x30x1x1xf32, 1x30x1x1xf32)
        add__48 = paddle._C_ops.add_(conv2d_39, reshape_78)

        # pd_op.shape: (4xi32) <- (-1x30x1x1xf32)
        shape_11 = paddle._C_ops.shape(add__48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_11, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_22 = [slice_11, full_78]

        # pd_op.reshape_: (-1x-1xf32, 0x-1x30x1x1xf32) <- (-1x30x1x1xf32, [1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))
        return reshape__44



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_7, parameter_4, parameter_6, parameter_5, parameter_8, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_163, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, feed_0):
        return self.builtin_module_629_0_0(parameter_0, parameter_1, parameter_2, parameter_3, parameter_7, parameter_4, parameter_6, parameter_5, parameter_8, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_47, parameter_44, parameter_46, parameter_45, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_83, parameter_87, parameter_84, parameter_86, parameter_85, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_121, parameter_122, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_163, parameter_167, parameter_164, parameter_166, parameter_165, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_629_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([192, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([192, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([192, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([192, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([384, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([384, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([384, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([768, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([768, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([768, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([25, 75, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([25], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([30, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 2, 350, 25, 1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[192, 2, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[192, 64, 1, 1], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[192, 64, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[192, 64, 1, 1], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[384, 64, 1, 1], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[384, 128, 1, 1], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[384, 128, 1, 1], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[768, 128, 1, 1], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[768, 256, 1, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[768, 256, 1, 1], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[25, 75, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[30, 256, 1, 1], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 2, 350, 25, 1], dtype='float32'),
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