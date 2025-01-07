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
    return [553][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_783_0_0(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_68, parameter_67, parameter_69, parameter_70, parameter_71, parameter_72, parameter_74, parameter_73, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_81, parameter_80, parameter_82, parameter_83, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_94, parameter_93, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_102, parameter_101, parameter_104, parameter_103, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_161, parameter_160, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_183, parameter_182, parameter_184, parameter_185, parameter_186, parameter_187, parameter_189, parameter_188, parameter_190, parameter_191, parameter_192, parameter_193, parameter_195, parameter_194, parameter_196, parameter_197, parameter_198, parameter_199, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_219, parameter_218, parameter_220, parameter_221, parameter_222, feed_0):

        # pd_op.bilinear_interp: (-1x3x32x64xf32) <- (-1x3x64x256xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(feed_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

        # pd_op.conv2d: (-1x32x32x64xf32) <- (-1x3x32x64xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(bilinear_interp_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x32x64xf32) <- (-1x32x32x64xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (-1x32x32x64xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x64xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x32x64xf32) <- (-1x32x32x64xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x32x16x32xf32) <- (-1x32x32x64xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, full_int_array_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x16x32xf32) <- (-1x32x16x32xf32, 64x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x16x32xf32) <- (-1x64x16x32xf32, 1x64x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.batch_norm_: (-1x64x16x32xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x16x32xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x16x32xf32) <- (-1x64x16x32xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 2]

        # pd_op.pool2d: (-1x64x8x16xf32) <- (-1x64x16x32xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__1, full_int_array_3, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x8x16xf32) <- (-1x64x8x16xf32, 128x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x8x16xf32) <- (-1x128x8x16xf32, 1x128x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_2, reshape_4)

        # pd_op.batch_norm_: (-1x128x8x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x8x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__2, parameter_14, parameter_15, parameter_16, parameter_17, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x8x16xf32) <- (-1x128x8x16xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [2, 2]

        # pd_op.pool2d: (-1x128x4x8xf32) <- (-1x128x8x16xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__2, full_int_array_5, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x4x8xf32) <- (-1x128x4x8xf32, 256x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_2, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x4x8xf32) <- (-1x256x4x8xf32, 1x256x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_3, reshape_6)

        # pd_op.batch_norm_: (-1x256x4x8xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x4x8xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_20, parameter_21, parameter_22, parameter_23, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x4x8xf32) <- (-1x256x4x8xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [2, 2]

        # pd_op.pool2d: (-1x256x2x4xf32) <- (-1x256x4x8xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__3, full_int_array_7, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x2x4xf32) <- (-1x256x2x4xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_3, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_25, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x2x4xf32) <- (-1x256x2x4xf32, 1x256x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_4, reshape_8)

        # pd_op.batch_norm_: (-1x256x2x4xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x2x4xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__4, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x2x4xf32) <- (-1x256x2x4xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [2, 2]

        # pd_op.pool2d: (-1x256x1x2xf32) <- (-1x256x2x4xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__4, full_int_array_9, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x2xf32) <- (-1x256x1x2xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_4, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_31, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x2xf32) <- (-1x256x1x2xf32, 1x256x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_5, reshape_10)

        # pd_op.batch_norm_: (-1x256x1x2xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x1x2xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__5, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x2xf32) <- (-1x256x1x2xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.shape: (4xi32) <- (-1x256x1x2xf32)
        shape_0 = paddle._C_ops.shape(relu__5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_0 = [slice_0, full_0]

        # pd_op.reshape_: (-1x-1xf32, 0x-1x256x1x2xf32) <- (-1x256x1x2xf32, [1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(relu__5, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x512xf32) <- (-1x-1xf32, 512x512xf32)
        matmul_0 = paddle._C_ops.matmul(reshape__0, parameter_36, False, False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__6 = paddle._C_ops.add_(matmul_0, parameter_37)

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__6, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512xf32) <- (-1x512xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x512xf32) <- (-1x512xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(relu__6, full_1, float('0'), True)

        # pd_op.matmul: (-1x40xf32) <- (-1x512xf32, 512x40xf32)
        matmul_1 = paddle._C_ops.matmul(scale__0, parameter_42, False, False)

        # pd_op.add_: (-1x40xf32) <- (-1x40xf32, 40xf32)
        add__7 = paddle._C_ops.add_(matmul_1, parameter_43)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_13 = [-1, 20, 2]

        # pd_op.reshape_: (-1x20x2xf32, 0x-1x40xf32) <- (-1x40xf32, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__7, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x20x2xf32)
        shape_1 = paddle._C_ops.shape(reshape__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_2, full_3]

        # pd_op.expand: (-1x3x2xf32) <- (3x2xf32, [1xi32, 1xi32, 1xi32])
        expand_0 = paddle._C_ops.expand(parameter_44, [x.reshape([]) for x in combine_1])

        # pd_op.cast_: (-1x20x2xf32) <- (-1x20x2xf32)
        cast__0 = paddle._C_ops.cast_(reshape__2, paddle.float32)

        # builtin.combine: ([-1x20x2xf32, -1x3x2xf32]) <- (-1x20x2xf32, -1x3x2xf32)
        combine_2 = [cast__0, expand_0]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x23x2xf32) <- ([-1x20x2xf32, -1x3x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_4)

        # pd_op.matmul: (-1x23x2xf32) <- (23x23xf32, -1x23x2xf32)
        matmul_2 = paddle._C_ops.matmul(parameter_45, concat_0, False, False)

        # pd_op.matmul: (-1x3200x2xf32) <- (3200x23xf32, -1x23x2xf32)
        matmul_3 = paddle._C_ops.matmul(parameter_46, matmul_2, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [-1, 32, 100, 2]

        # pd_op.reshape_: (-1x32x100x2xf32, 0x-1x3200x2xf32) <- (-1x3200x2xf32, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_3, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x32x100x2xf32) <- (-1x32x100x2xf32, 1xf32, 1xf32)
        clip__0 = paddle._C_ops.clip_(reshape__4, full_5, full_6)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x32x100x2xf32) <- (-1x32x100x2xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(clip__0, full_7, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x32x100x2xf32) <- (-1x32x100x2xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(scale__1, full_8, float('-1'), True)

        # pd_op.grid_sample: (-1x3x32x100xf32) <- (-1x3x64x256xf32, -1x32x100x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(feed_0, scale__2, 'bilinear', 'zeros', True)

        # pd_op.conv2d: (-1x32x16x50xf32) <- (-1x3x32x100xf32, 32x3x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(grid_sample_0, parameter_47, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_48, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x16x50xf32) <- (-1x32x16x50xf32, 1x32x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_6, reshape_12)

        # pd_op.batch_norm_: (-1x32x16x50xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x50xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__8, parameter_49, parameter_50, parameter_51, parameter_52, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.gelu: (-1x32x16x50xf32) <- (-1x32x16x50xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__42, False)

        # pd_op.conv2d: (-1x64x8x25xf32) <- (-1x32x16x50xf32, 64x32x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(gelu_0, parameter_53, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_54, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x8x25xf32) <- (-1x64x8x25xf32, 1x64x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_7, reshape_14)

        # pd_op.batch_norm_: (-1x64x8x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x8x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__9, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.gelu: (-1x64x8x25xf32) <- (-1x64x8x25xf32)
        gelu_1 = paddle._C_ops.gelu(batch_norm__48, False)

        # pd_op.flatten_: (-1x64x200xf32, None) <- (-1x64x8x25xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(gelu_1, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x200x64xf32) <- (-1x64x200xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, 1x200x64xf32)
        add__10 = paddle._C_ops.add_(transpose_0, parameter_59)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__10, parameter_60, parameter_61, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x192xf32) <- (-1x200x64xf32, 64x192xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_0, parameter_62, False, False)

        # pd_op.add_: (-1x200x192xf32) <- (-1x200x192xf32, 192xf32)
        add__11 = paddle._C_ops.add_(matmul_4, parameter_63)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_19 = [0, -1, 3, 2, 32]

        # pd_op.reshape_: (-1x-1x3x2x32xf32, 0x-1x200x192xf32) <- (-1x200x192xf32, 5xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x2x-1x32xf32) <- (-1x-1x3x2x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_1, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x2x-1x32xf32) <- (-1x2x-1x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_2, full_9, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [2]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_1, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [3]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_1, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.transpose: (-1x2x32x-1xf32) <- (-1x2x-1x32xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x-1x-1xf32) <- (-1x2x-1x32xf32, -1x2x32x-1xf32)
        matmul_5 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)

        # pd_op.add: (-1x2x200x200xf32) <- (-1x2x-1x-1xf32, 1x1x200x200xf32)
        add_0 = paddle._C_ops.add(matmul_5, parameter_64)

        # pd_op.softmax_: (-1x2x200x200xf32) <- (-1x2x200x200xf32)
        softmax__0 = paddle._C_ops.softmax_(add_0, -1)

        # pd_op.matmul: (-1x2x200x32xf32) <- (-1x2x200x200xf32, -1x2x-1x32xf32)
        matmul_6 = paddle._C_ops.matmul(softmax__0, slice_4, False, False)

        # pd_op.transpose: (-1x200x2x32xf32) <- (-1x2x200x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_26 = [0, -1, 64]

        # pd_op.reshape_: (-1x-1x64xf32, 0x-1x200x2x32xf32) <- (-1x200x2x32xf32, 3xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_7 = paddle._C_ops.matmul(reshape__8, parameter_65, False, False)

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_1 = paddle._C_ops.add(matmul_7, parameter_66)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x-1x64xf32)
        add__12 = paddle._C_ops.add_(add__10, add_1)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__12, parameter_67, parameter_68, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x256xf32) <- (-1x200x64xf32, 64x256xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_3, parameter_69, False, False)

        # pd_op.add_: (-1x200x256xf32) <- (-1x200x256xf32, 256xf32)
        add__13 = paddle._C_ops.add_(matmul_8, parameter_70)

        # pd_op.gelu: (-1x200x256xf32) <- (-1x200x256xf32)
        gelu_2 = paddle._C_ops.gelu(add__13, False)

        # pd_op.matmul: (-1x200x64xf32) <- (-1x200x256xf32, 256x64xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_2, parameter_71, False, False)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, 64xf32)
        add__14 = paddle._C_ops.add_(matmul_9, parameter_72)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x200x64xf32)
        add__15 = paddle._C_ops.add_(add__12, add__14)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__15, parameter_73, parameter_74, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x192xf32) <- (-1x200x64xf32, 64x192xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_75, False, False)

        # pd_op.add_: (-1x200x192xf32) <- (-1x200x192xf32, 192xf32)
        add__16 = paddle._C_ops.add_(matmul_10, parameter_76)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_27 = [0, -1, 3, 2, 32]

        # pd_op.reshape_: (-1x-1x3x2x32xf32, 0x-1x200x192xf32) <- (-1x200x192xf32, 5xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__16, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x2x-1x32xf32) <- (-1x-1x3x2x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__10, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_4, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x2x-1x32xf32) <- (-1x2x-1x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_5, full_10, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_4, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [3]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_4, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.transpose: (-1x2x32x-1xf32) <- (-1x2x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x-1x-1xf32) <- (-1x2x-1x32xf32, -1x2x32x-1xf32)
        matmul_11 = paddle._C_ops.matmul(scale_1, transpose_5, False, False)

        # pd_op.add: (-1x2x200x200xf32) <- (-1x2x-1x-1xf32, 1x1x200x200xf32)
        add_2 = paddle._C_ops.add(matmul_11, parameter_77)

        # pd_op.softmax_: (-1x2x200x200xf32) <- (-1x2x200x200xf32)
        softmax__1 = paddle._C_ops.softmax_(add_2, -1)

        # pd_op.matmul: (-1x2x200x32xf32) <- (-1x2x200x200xf32, -1x2x-1x32xf32)
        matmul_12 = paddle._C_ops.matmul(softmax__1, slice_7, False, False)

        # pd_op.transpose: (-1x200x2x32xf32) <- (-1x2x200x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_34 = [0, -1, 64]

        # pd_op.reshape_: (-1x-1x64xf32, 0x-1x200x2x32xf32) <- (-1x200x2x32xf32, 3xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_13 = paddle._C_ops.matmul(reshape__12, parameter_78, False, False)

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_3 = paddle._C_ops.add(matmul_13, parameter_79)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x-1x64xf32)
        add__17 = paddle._C_ops.add_(add__15, add_3)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__17, parameter_80, parameter_81, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x256xf32) <- (-1x200x64xf32, 64x256xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_82, False, False)

        # pd_op.add_: (-1x200x256xf32) <- (-1x200x256xf32, 256xf32)
        add__18 = paddle._C_ops.add_(matmul_14, parameter_83)

        # pd_op.gelu: (-1x200x256xf32) <- (-1x200x256xf32)
        gelu_3 = paddle._C_ops.gelu(add__18, False)

        # pd_op.matmul: (-1x200x64xf32) <- (-1x200x256xf32, 256x64xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_3, parameter_84, False, False)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, 64xf32)
        add__19 = paddle._C_ops.add_(matmul_15, parameter_85)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x200x64xf32)
        add__20 = paddle._C_ops.add_(add__17, add__19)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__20, parameter_86, parameter_87, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x192xf32) <- (-1x200x64xf32, 64x192xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_88, False, False)

        # pd_op.add_: (-1x200x192xf32) <- (-1x200x192xf32, 192xf32)
        add__21 = paddle._C_ops.add_(matmul_16, parameter_89)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_35 = [0, -1, 3, 2, 32]

        # pd_op.reshape_: (-1x-1x3x2x32xf32, 0x-1x200x192xf32) <- (-1x200x192xf32, 5xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__21, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x2x-1x32xf32) <- (-1x-1x3x2x32xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__14, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_7, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x2x-1x32xf32) <- (-1x2x-1x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_8, full_11, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_7, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [3]

        # pd_op.slice: (-1x2x-1x32xf32) <- (3x-1x2x-1x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_7, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.transpose: (-1x2x32x-1xf32) <- (-1x2x-1x32xf32)
        transpose_8 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x-1x-1xf32) <- (-1x2x-1x32xf32, -1x2x32x-1xf32)
        matmul_17 = paddle._C_ops.matmul(scale_2, transpose_8, False, False)

        # pd_op.add: (-1x2x200x200xf32) <- (-1x2x-1x-1xf32, 1x1x200x200xf32)
        add_4 = paddle._C_ops.add(matmul_17, parameter_90)

        # pd_op.softmax_: (-1x2x200x200xf32) <- (-1x2x200x200xf32)
        softmax__2 = paddle._C_ops.softmax_(add_4, -1)

        # pd_op.matmul: (-1x2x200x32xf32) <- (-1x2x200x200xf32, -1x2x-1x32xf32)
        matmul_18 = paddle._C_ops.matmul(softmax__2, slice_10, False, False)

        # pd_op.transpose: (-1x200x2x32xf32) <- (-1x2x200x32xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_42 = [0, -1, 64]

        # pd_op.reshape_: (-1x-1x64xf32, 0x-1x200x2x32xf32) <- (-1x200x2x32xf32, 3xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_19 = paddle._C_ops.matmul(reshape__16, parameter_91, False, False)

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_5 = paddle._C_ops.add(matmul_19, parameter_92)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x-1x64xf32)
        add__22 = paddle._C_ops.add_(add__20, add_5)

        # pd_op.layer_norm: (-1x200x64xf32, -200xf32, -200xf32) <- (-1x200x64xf32, 64xf32, 64xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_93, parameter_94, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x200x256xf32) <- (-1x200x64xf32, 64x256xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_15, parameter_95, False, False)

        # pd_op.add_: (-1x200x256xf32) <- (-1x200x256xf32, 256xf32)
        add__23 = paddle._C_ops.add_(matmul_20, parameter_96)

        # pd_op.gelu: (-1x200x256xf32) <- (-1x200x256xf32)
        gelu_4 = paddle._C_ops.gelu(add__23, False)

        # pd_op.matmul: (-1x200x64xf32) <- (-1x200x256xf32, 256x64xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_4, parameter_97, False, False)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, 64xf32)
        add__24 = paddle._C_ops.add_(matmul_21, parameter_98)

        # pd_op.add_: (-1x200x64xf32) <- (-1x200x64xf32, -1x200x64xf32)
        add__25 = paddle._C_ops.add_(add__22, add__24)

        # pd_op.transpose: (-1x64x200xf32) <- (-1x200x64xf32)
        transpose_10 = paddle._C_ops.transpose(add__25, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [0, 64, 8, 25]

        # pd_op.reshape_: (-1x64x8x25xf32, 0x-1x64x200xf32) <- (-1x64x200xf32, 4xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x4x25xf32) <- (-1x64x8x25xf32, 128x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(reshape__18, parameter_99, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_100, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x25xf32) <- (-1x128x4x25xf32, 1x128x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_8, reshape_16)

        # pd_op.flatten_: (-1x128x100xf32, None) <- (-1x128x4x25xf32)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__26, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x100x128xf32) <- (-1x128x100xf32)
        transpose_11 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_11, parameter_101, parameter_102, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_18, parameter_103, parameter_104, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_21, parameter_105, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__27 = paddle._C_ops.add_(matmul_22, parameter_106)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_45 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__27, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__20, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_12, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_11, full_12, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_12, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_12, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(slice_12, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_23 = paddle._C_ops.matmul(scale_3, transpose_13, False, False)

        # pd_op.add: (-1x4x100x100xf32) <- (-1x4x-1x-1xf32, 1x1x100x100xf32)
        add_6 = paddle._C_ops.add(matmul_23, parameter_107)

        # pd_op.softmax_: (-1x4x100x100xf32) <- (-1x4x100x100xf32)
        softmax__3 = paddle._C_ops.softmax_(add_6, -1)

        # pd_op.matmul: (-1x4x100x32xf32) <- (-1x4x100x100xf32, -1x4x-1x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax__3, slice_13, False, False)

        # pd_op.transpose: (-1x100x4x32xf32) <- (-1x4x100x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_52 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x100x4x32xf32) <- (-1x100x4x32xf32, 3xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_25 = paddle._C_ops.matmul(reshape__22, parameter_108, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_7 = paddle._C_ops.add(matmul_25, parameter_109)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__28 = paddle._C_ops.add_(layer_norm_18, add_7)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_110, parameter_111, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_24, parameter_112, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__29 = paddle._C_ops.add_(matmul_26, parameter_113)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_5 = paddle._C_ops.gelu(add__29, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_5, parameter_114, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__30 = paddle._C_ops.add_(matmul_27, parameter_115)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__31 = paddle._C_ops.add_(add__28, add__30)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__31, parameter_116, parameter_117, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_27, parameter_118, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__32 = paddle._C_ops.add_(matmul_28, parameter_119)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_53 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__32, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__24, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_15, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_14, full_13, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_15, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_15, [0], full_int_array_58, full_int_array_59, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_29 = paddle._C_ops.matmul(scale_4, transpose_16, False, False)

        # pd_op.add: (-1x4x100x100xf32) <- (-1x4x-1x-1xf32, 1x1x100x100xf32)
        add_8 = paddle._C_ops.add(matmul_29, parameter_120)

        # pd_op.softmax_: (-1x4x100x100xf32) <- (-1x4x100x100xf32)
        softmax__4 = paddle._C_ops.softmax_(add_8, -1)

        # pd_op.matmul: (-1x4x100x32xf32) <- (-1x4x100x100xf32, -1x4x-1x32xf32)
        matmul_30 = paddle._C_ops.matmul(softmax__4, slice_16, False, False)

        # pd_op.transpose: (-1x100x4x32xf32) <- (-1x4x100x32xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_30, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_60 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x100x4x32xf32) <- (-1x100x4x32xf32, 3xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_31 = paddle._C_ops.matmul(reshape__26, parameter_121, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_9 = paddle._C_ops.add(matmul_31, parameter_122)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__33 = paddle._C_ops.add_(add__31, add_9)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_123, parameter_124, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_30, parameter_125, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__34 = paddle._C_ops.add_(matmul_32, parameter_126)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_6 = paddle._C_ops.gelu(add__34, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_33 = paddle._C_ops.matmul(gelu_6, parameter_127, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__35 = paddle._C_ops.add_(matmul_33, parameter_128)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__36 = paddle._C_ops.add_(add__33, add__35)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__36, parameter_129, parameter_130, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_33, parameter_131, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__37 = paddle._C_ops.add_(matmul_34, parameter_132)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_61 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__37, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_18, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_17, full_14, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_18, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_18, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_19 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_35 = paddle._C_ops.matmul(scale_5, transpose_19, False, False)

        # pd_op.add: (-1x4x100x100xf32) <- (-1x4x-1x-1xf32, 1x1x100x100xf32)
        add_10 = paddle._C_ops.add(matmul_35, parameter_133)

        # pd_op.softmax_: (-1x4x100x100xf32) <- (-1x4x100x100xf32)
        softmax__5 = paddle._C_ops.softmax_(add_10, -1)

        # pd_op.matmul: (-1x4x100x32xf32) <- (-1x4x100x100xf32, -1x4x-1x32xf32)
        matmul_36 = paddle._C_ops.matmul(softmax__5, slice_19, False, False)

        # pd_op.transpose: (-1x100x4x32xf32) <- (-1x4x100x32xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_68 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x100x4x32xf32) <- (-1x100x4x32xf32, 3xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_37 = paddle._C_ops.matmul(reshape__30, parameter_134, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_11 = paddle._C_ops.add(matmul_37, parameter_135)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__38 = paddle._C_ops.add_(add__36, add_11)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__38, parameter_136, parameter_137, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_36, parameter_138, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__39 = paddle._C_ops.add_(matmul_38, parameter_139)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_7 = paddle._C_ops.gelu(add__39, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_7, parameter_140, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__40 = paddle._C_ops.add_(matmul_39, parameter_141)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__41 = paddle._C_ops.add_(add__38, add__40)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__41, parameter_142, parameter_143, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_39, parameter_144, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__42 = paddle._C_ops.add_(matmul_40, parameter_145)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_69 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__42, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__32, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_21, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_20, full_15, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_21, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_21, [0], full_int_array_74, full_int_array_75, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_22 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_41 = paddle._C_ops.matmul(scale_6, transpose_22, False, False)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(matmul_41, -1)

        # pd_op.matmul: (-1x4x-1x32xf32) <- (-1x4x-1x-1xf32, -1x4x-1x32xf32)
        matmul_42 = paddle._C_ops.matmul(softmax_0, slice_22, False, False)

        # pd_op.transpose: (-1x-1x4x32xf32) <- (-1x4x-1x32xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_42, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_76 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x-1x4x32xf32) <- (-1x-1x4x32xf32, 3xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_43 = paddle._C_ops.matmul(reshape__34, parameter_146, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_12 = paddle._C_ops.add(matmul_43, parameter_147)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__43 = paddle._C_ops.add_(add__41, add_12)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_148, parameter_149, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_44 = paddle._C_ops.matmul(layer_norm_42, parameter_150, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__44 = paddle._C_ops.add_(matmul_44, parameter_151)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_8 = paddle._C_ops.gelu(add__44, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_45 = paddle._C_ops.matmul(gelu_8, parameter_152, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__45 = paddle._C_ops.add_(matmul_45, parameter_153)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__46 = paddle._C_ops.add_(add__43, add__45)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__46, parameter_154, parameter_155, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_45, parameter_156, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__47 = paddle._C_ops.add_(matmul_46, parameter_157)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_77 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__47, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_24 = paddle._C_ops.transpose(reshape__36, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_24, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_23, full_16, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_24, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_24, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_25 = paddle._C_ops.transpose(slice_24, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_47 = paddle._C_ops.matmul(scale_7, transpose_25, False, False)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(matmul_47, -1)

        # pd_op.matmul: (-1x4x-1x32xf32) <- (-1x4x-1x-1xf32, -1x4x-1x32xf32)
        matmul_48 = paddle._C_ops.matmul(softmax_1, slice_25, False, False)

        # pd_op.transpose: (-1x-1x4x32xf32) <- (-1x4x-1x32xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_48, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_84 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x-1x4x32xf32) <- (-1x-1x4x32xf32, 3xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_49 = paddle._C_ops.matmul(reshape__38, parameter_158, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_13 = paddle._C_ops.add(matmul_49, parameter_159)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__48 = paddle._C_ops.add_(add__46, add_13)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__48, parameter_160, parameter_161, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_48, parameter_162, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__49 = paddle._C_ops.add_(matmul_50, parameter_163)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_9 = paddle._C_ops.gelu(add__49, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_51 = paddle._C_ops.matmul(gelu_9, parameter_164, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__50 = paddle._C_ops.add_(matmul_51, parameter_165)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__51 = paddle._C_ops.add_(add__48, add__50)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__51, parameter_166, parameter_167, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x384xf32) <- (-1x100x128xf32, 128x384xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_51, parameter_168, False, False)

        # pd_op.add_: (-1x100x384xf32) <- (-1x100x384xf32, 384xf32)
        add__52 = paddle._C_ops.add_(matmul_52, parameter_169)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_85 = [0, -1, 3, 4, 32]

        # pd_op.reshape_: (-1x-1x3x4x32xf32, 0x-1x100x384xf32) <- (-1x100x384xf32, 5xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__52, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x-1x32xf32) <- (-1x-1x3x4x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [1]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_27, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x32xf32) <- (-1x4x-1x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_26, full_17, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [2]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_27, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [3]

        # pd_op.slice: (-1x4x-1x32xf32) <- (3x-1x4x-1x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(transpose_27, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.transpose: (-1x4x32x-1xf32) <- (-1x4x-1x32xf32)
        transpose_28 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x-1x-1xf32) <- (-1x4x-1x32xf32, -1x4x32x-1xf32)
        matmul_53 = paddle._C_ops.matmul(scale_8, transpose_28, False, False)

        # pd_op.softmax: (-1x4x-1x-1xf32) <- (-1x4x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(matmul_53, -1)

        # pd_op.matmul: (-1x4x-1x32xf32) <- (-1x4x-1x-1xf32, -1x4x-1x32xf32)
        matmul_54 = paddle._C_ops.matmul(softmax_2, slice_28, False, False)

        # pd_op.transpose: (-1x-1x4x32xf32) <- (-1x4x-1x32xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_54, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_92 = [0, -1, 128]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x-1x4x32xf32) <- (-1x-1x4x32xf32, 3xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_55 = paddle._C_ops.matmul(reshape__42, parameter_170, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_14 = paddle._C_ops.add(matmul_55, parameter_171)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x-1x128xf32)
        add__53 = paddle._C_ops.add_(add__51, add_14)

        # pd_op.layer_norm: (-1x100x128xf32, -100xf32, -100xf32) <- (-1x100x128xf32, 128xf32, 128xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__53, parameter_172, parameter_173, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x100x512xf32) <- (-1x100x128xf32, 128x512xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_54, parameter_174, False, False)

        # pd_op.add_: (-1x100x512xf32) <- (-1x100x512xf32, 512xf32)
        add__54 = paddle._C_ops.add_(matmul_56, parameter_175)

        # pd_op.gelu: (-1x100x512xf32) <- (-1x100x512xf32)
        gelu_10 = paddle._C_ops.gelu(add__54, False)

        # pd_op.matmul: (-1x100x128xf32) <- (-1x100x512xf32, 512x128xf32)
        matmul_57 = paddle._C_ops.matmul(gelu_10, parameter_176, False, False)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, 128xf32)
        add__55 = paddle._C_ops.add_(matmul_57, parameter_177)

        # pd_op.add_: (-1x100x128xf32) <- (-1x100x128xf32, -1x100x128xf32)
        add__56 = paddle._C_ops.add_(add__53, add__55)

        # pd_op.transpose: (-1x128x100xf32) <- (-1x100x128xf32)
        transpose_30 = paddle._C_ops.transpose(add__56, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_93 = [0, 128, 4, 25]

        # pd_op.reshape_: (-1x128x4x25xf32, 0x-1x128x100xf32) <- (-1x128x100xf32, 4xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x2x25xf32) <- (-1x128x4x25xf32, 256x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(reshape__44, parameter_178, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_94 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_179, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x2x25xf32) <- (-1x256x2x25xf32, 1x256x1x1xf32)
        add__57 = paddle._C_ops.add_(conv2d_9, reshape_18)

        # pd_op.flatten_: (-1x256x50xf32, None) <- (-1x256x2x25xf32)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__57, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x50x256xf32) <- (-1x256x50xf32)
        transpose_31 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_31, parameter_180, parameter_181, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_57, parameter_182, parameter_183, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x768xf32) <- (-1x50x256xf32, 256x768xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_60, parameter_184, False, False)

        # pd_op.add_: (-1x50x768xf32) <- (-1x50x768xf32, 768xf32)
        add__58 = paddle._C_ops.add_(matmul_58, parameter_185)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_95 = [0, -1, 3, 8, 32]

        # pd_op.reshape_: (-1x-1x3x8x32xf32, 0x-1x50x768xf32) <- (-1x50x768xf32, 5xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__58, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_32 = paddle._C_ops.transpose(reshape__46, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [1]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_32, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x32xf32) <- (-1x8x-1x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_29, full_18, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [2]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_32, [0], full_int_array_98, full_int_array_99, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [3]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_32, [0], full_int_array_100, full_int_array_101, [1], [0])

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_33 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_59 = paddle._C_ops.matmul(scale_9, transpose_33, False, False)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(matmul_59, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_60 = paddle._C_ops.matmul(softmax_3, slice_31, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_34 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_102 = [0, -1, 256]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x-1x8x32xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_34, full_int_array_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_61 = paddle._C_ops.matmul(reshape__48, parameter_186, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_61, parameter_187)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x-1x256xf32)
        add__59 = paddle._C_ops.add_(layer_norm_57, add_15)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__59, parameter_188, parameter_189, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x1024xf32) <- (-1x50x256xf32, 256x1024xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_63, parameter_190, False, False)

        # pd_op.add_: (-1x50x1024xf32) <- (-1x50x1024xf32, 1024xf32)
        add__60 = paddle._C_ops.add_(matmul_62, parameter_191)

        # pd_op.gelu: (-1x50x1024xf32) <- (-1x50x1024xf32)
        gelu_11 = paddle._C_ops.gelu(add__60, False)

        # pd_op.matmul: (-1x50x256xf32) <- (-1x50x1024xf32, 1024x256xf32)
        matmul_63 = paddle._C_ops.matmul(gelu_11, parameter_192, False, False)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, 256xf32)
        add__61 = paddle._C_ops.add_(matmul_63, parameter_193)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x50x256xf32)
        add__62 = paddle._C_ops.add_(add__59, add__61)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__62, parameter_194, parameter_195, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x768xf32) <- (-1x50x256xf32, 256x768xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_66, parameter_196, False, False)

        # pd_op.add_: (-1x50x768xf32) <- (-1x50x768xf32, 768xf32)
        add__63 = paddle._C_ops.add_(matmul_64, parameter_197)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_103 = [0, -1, 3, 8, 32]

        # pd_op.reshape_: (-1x-1x3x8x32xf32, 0x-1x50x768xf32) <- (-1x50x768xf32, 5xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__63, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_35 = paddle._C_ops.transpose(reshape__50, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [1]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_35, [0], full_int_array_104, full_int_array_105, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x32xf32) <- (-1x8x-1x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_32, full_19, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [2]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_35, [0], full_int_array_106, full_int_array_107, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [3]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_35, [0], full_int_array_108, full_int_array_109, [1], [0])

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_36 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_65 = paddle._C_ops.matmul(scale_10, transpose_36, False, False)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(matmul_65, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_66 = paddle._C_ops.matmul(softmax_4, slice_34, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_37 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_110 = [0, -1, 256]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x-1x8x32xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, full_int_array_110), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_67 = paddle._C_ops.matmul(reshape__52, parameter_198, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_16 = paddle._C_ops.add(matmul_67, parameter_199)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x-1x256xf32)
        add__64 = paddle._C_ops.add_(add__62, add_16)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__64, parameter_200, parameter_201, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x1024xf32) <- (-1x50x256xf32, 256x1024xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_69, parameter_202, False, False)

        # pd_op.add_: (-1x50x1024xf32) <- (-1x50x1024xf32, 1024xf32)
        add__65 = paddle._C_ops.add_(matmul_68, parameter_203)

        # pd_op.gelu: (-1x50x1024xf32) <- (-1x50x1024xf32)
        gelu_12 = paddle._C_ops.gelu(add__65, False)

        # pd_op.matmul: (-1x50x256xf32) <- (-1x50x1024xf32, 1024x256xf32)
        matmul_69 = paddle._C_ops.matmul(gelu_12, parameter_204, False, False)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, 256xf32)
        add__66 = paddle._C_ops.add_(matmul_69, parameter_205)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x50x256xf32)
        add__67 = paddle._C_ops.add_(add__64, add__66)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__67, parameter_206, parameter_207, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x768xf32) <- (-1x50x256xf32, 256x768xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_72, parameter_208, False, False)

        # pd_op.add_: (-1x50x768xf32) <- (-1x50x768xf32, 768xf32)
        add__68 = paddle._C_ops.add_(matmul_70, parameter_209)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_111 = [0, -1, 3, 8, 32]

        # pd_op.reshape_: (-1x-1x3x8x32xf32, 0x-1x50x768xf32) <- (-1x50x768xf32, 5xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__68, full_int_array_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_38 = paddle._C_ops.transpose(reshape__54, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_38, [0], full_int_array_112, full_int_array_113, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x32xf32) <- (-1x8x-1x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_35, full_20, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [2]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_38, [0], full_int_array_114, full_int_array_115, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [3]

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_38, [0], full_int_array_116, full_int_array_117, [1], [0])

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_39 = paddle._C_ops.transpose(slice_36, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_71 = paddle._C_ops.matmul(scale_11, transpose_39, False, False)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(matmul_71, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_72 = paddle._C_ops.matmul(softmax_5, slice_37, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_40 = paddle._C_ops.transpose(matmul_72, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_118 = [0, -1, 256]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x-1x8x32xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_40, full_int_array_118), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_73 = paddle._C_ops.matmul(reshape__56, parameter_210, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_17 = paddle._C_ops.add(matmul_73, parameter_211)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x-1x256xf32)
        add__69 = paddle._C_ops.add_(add__67, add_17)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__69, parameter_212, parameter_213, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x50x1024xf32) <- (-1x50x256xf32, 256x1024xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_75, parameter_214, False, False)

        # pd_op.add_: (-1x50x1024xf32) <- (-1x50x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add_(matmul_74, parameter_215)

        # pd_op.gelu: (-1x50x1024xf32) <- (-1x50x1024xf32)
        gelu_13 = paddle._C_ops.gelu(add__70, False)

        # pd_op.matmul: (-1x50x256xf32) <- (-1x50x1024xf32, 1024x256xf32)
        matmul_75 = paddle._C_ops.matmul(gelu_13, parameter_216, False, False)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, 256xf32)
        add__71 = paddle._C_ops.add_(matmul_75, parameter_217)

        # pd_op.add_: (-1x50x256xf32) <- (-1x50x256xf32, -1x50x256xf32)
        add__72 = paddle._C_ops.add_(add__69, add__71)

        # pd_op.layer_norm: (-1x50x256xf32, -50xf32, -50xf32) <- (-1x50x256xf32, 256xf32, 256xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__72, parameter_218, parameter_219, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.transpose: (-1x256x50xf32) <- (-1x50x256xf32)
        transpose_41 = paddle._C_ops.transpose(layer_norm_78, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_119 = [0, 256, 2, 25]

        # pd_op.reshape_: (-1x256x2x25xf32, 0x-1x256x50xf32) <- (-1x256x50xf32, 4xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_41, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_120 = [1, 25]

        # pd_op.pool2d: (-1x256x1x25xf32) <- (-1x256x2x25xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(reshape__58, full_int_array_120, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x192x1x25xf32) <- (-1x256x1x25xf32, 192x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_5, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.hardswish: (-1x192x1x25xf32) <- (-1x192x1x25xf32)
        hardswish_0 = paddle._C_ops.hardswish(conv2d_10)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x192x1x25xf32, None) <- (-1x192x1x25xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(hardswish_0, None, full_21, True, 'downgrade_in_infer', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [2]

        # pd_op.squeeze_: (-1x192x25xf32, None) <- (-1x192x1x25xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(dropout_0, full_int_array_121), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x25x192xf32) <- (-1x192x25xf32)
        transpose_42 = paddle._C_ops.transpose(squeeze__0, [0, 2, 1])

        # pd_op.matmul: (-1x25x37xf32) <- (-1x25x192xf32, 192x37xf32)
        matmul_76 = paddle._C_ops.matmul(transpose_42, parameter_221, False, False)

        # pd_op.add_: (-1x25x37xf32) <- (-1x25x37xf32, 37xf32)
        add__73 = paddle._C_ops.add_(matmul_76, parameter_222)

        # pd_op.softmax_: (-1x25x37xf32) <- (-1x25x37xf32)
        softmax__6 = paddle._C_ops.softmax_(add__73, 2)
        return softmax__6



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

    def forward(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_68, parameter_67, parameter_69, parameter_70, parameter_71, parameter_72, parameter_74, parameter_73, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_81, parameter_80, parameter_82, parameter_83, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_94, parameter_93, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_102, parameter_101, parameter_104, parameter_103, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_161, parameter_160, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_183, parameter_182, parameter_184, parameter_185, parameter_186, parameter_187, parameter_189, parameter_188, parameter_190, parameter_191, parameter_192, parameter_193, parameter_195, parameter_194, parameter_196, parameter_197, parameter_198, parameter_199, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_219, parameter_218, parameter_220, parameter_221, parameter_222, feed_0):
        return self.builtin_module_783_0_0(parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_13, parameter_17, parameter_14, parameter_16, parameter_15, parameter_18, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_61, parameter_60, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_68, parameter_67, parameter_69, parameter_70, parameter_71, parameter_72, parameter_74, parameter_73, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_81, parameter_80, parameter_82, parameter_83, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_94, parameter_93, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_102, parameter_101, parameter_104, parameter_103, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_113, parameter_114, parameter_115, parameter_117, parameter_116, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_137, parameter_136, parameter_138, parameter_139, parameter_140, parameter_141, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_161, parameter_160, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_183, parameter_182, parameter_184, parameter_185, parameter_186, parameter_187, parameter_189, parameter_188, parameter_190, parameter_191, parameter_192, parameter_193, parameter_195, parameter_194, parameter_196, parameter_197, parameter_198, parameter_199, parameter_201, parameter_200, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_219, parameter_218, parameter_220, parameter_221, parameter_222, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_783_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([512, 40], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([3, 2], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([23, 23], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([3200, 23], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([1, 200, 64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([1, 1, 200, 200], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64, 256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256, 64], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([1, 1, 200, 200], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64, 256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256, 64], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([1, 1, 200, 200], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64, 256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256, 64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([1, 1, 100, 100], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([1, 1, 100, 100], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([1, 1, 100, 100], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128, 512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512, 128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256, 1024], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([256, 1024], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([256, 1024], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([192, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([37], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 64, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[512, 40], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[3, 2], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[23, 23], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[3200, 23], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[1, 200, 64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[1, 1, 200, 200], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64, 256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256, 64], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[1, 1, 200, 200], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64, 256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256, 64], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[1, 1, 200, 200], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64, 256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256, 64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[1, 1, 100, 100], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[1, 1, 100, 100], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[1, 1, 100, 100], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[128, 512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512, 128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256, 1024], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[256, 1024], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[256, 1024], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[192, 256, 1, 1], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[37], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype='float32'),
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