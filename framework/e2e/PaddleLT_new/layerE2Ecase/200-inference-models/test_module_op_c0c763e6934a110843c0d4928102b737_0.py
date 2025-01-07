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
    return [454][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2105_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_240, feed_0):

        # pd_op.cast: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x16x-1x-1xf16) <- (-1x3x-1x-1xf16, 16x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x-1x-1xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x-1x-1xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x16x-1x-1xf16) <- (-1x16x-1x-1xf16, 16x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(batch_norm__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf16, 0x16xf16) <- (16xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_6, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x16x-1x-1xf16) <- (-1x16x-1x-1xf16, 1x16x1x1xf16)
        add_0 = paddle._C_ops.add(depthwise_conv2d_0, reshape_0)

        # pd_op.multiply: (-1x16x-1x-1xf16) <- (1xf16, -1x16x-1x-1xf16)
        multiply_0 = paddle._C_ops.multiply(parameter_7, add_0)

        # pd_op.add: (-1x16x-1x-1xf16) <- (-1x16x-1x-1xf16, 1xf16)
        add_1 = paddle._C_ops.add(multiply_0, parameter_8)

        # pd_op.hardswish: (-1x16x-1x-1xf16) <- (-1x16x-1x-1xf16)
        hardswish_0 = paddle._C_ops.hardswish(add_1)

        # pd_op.multiply: (-1x16x-1x-1xf16) <- (1xf16, -1x16x-1x-1xf16)
        multiply_1 = paddle._C_ops.multiply(parameter_9, hardswish_0)

        # pd_op.add: (-1x16x-1x-1xf16) <- (-1x16x-1x-1xf16, 1xf16)
        add_2 = paddle._C_ops.add(multiply_1, parameter_10)

        # pd_op.conv2d: (-1x32x-1x-1xf16) <- (-1x16x-1x-1xf16, 32x16x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(add_2, parameter_11, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_12, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 1x32x1x1xf16)
        add_3 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.multiply: (-1x32x-1x-1xf16) <- (1xf16, -1x32x-1x-1xf16)
        multiply_2 = paddle._C_ops.multiply(parameter_13, add_3)

        # pd_op.add: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 1xf16)
        add_4 = paddle._C_ops.add(multiply_2, parameter_14)

        # pd_op.hardswish: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16)
        hardswish_1 = paddle._C_ops.hardswish(add_4)

        # pd_op.multiply: (-1x32x-1x-1xf16) <- (1xf16, -1x32x-1x-1xf16)
        multiply_3 = paddle._C_ops.multiply(parameter_15, hardswish_1)

        # pd_op.add: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 1xf16)
        add_5 = paddle._C_ops.add(multiply_3, parameter_16)

        # pd_op.depthwise_conv2d: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 32x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(add_5, parameter_17, [2, 2], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_18, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 1x32x1x1xf16)
        add_6 = paddle._C_ops.add(depthwise_conv2d_1, reshape_4)

        # pd_op.multiply: (-1x32x-1x-1xf16) <- (1xf16, -1x32x-1x-1xf16)
        multiply_4 = paddle._C_ops.multiply(parameter_19, add_6)

        # pd_op.add: (-1x32x-1x-1xf16) <- (-1x32x-1x-1xf16, 1xf16)
        add_7 = paddle._C_ops.add(multiply_4, parameter_20)

        # pd_op.conv2d: (-1x48x-1x-1xf16) <- (-1x32x-1x-1xf16, 48x32x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(add_7, parameter_21, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf16, 0x48xf16) <- (48xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_22, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1x48x1x1xf16)
        add_8 = paddle._C_ops.add(conv2d_2, reshape_6)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_5 = paddle._C_ops.multiply(parameter_23, add_8)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_9 = paddle._C_ops.add(multiply_5, parameter_24)

        # pd_op.hardswish: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16)
        hardswish_2 = paddle._C_ops.hardswish(add_9)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_6 = paddle._C_ops.multiply(parameter_25, hardswish_2)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_10 = paddle._C_ops.add(multiply_6, parameter_26)

        # pd_op.depthwise_conv2d: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 48x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(add_10, parameter_27, [1, 1], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf16, 0x48xf16) <- (48xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_28, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1x48x1x1xf16)
        add_11 = paddle._C_ops.add(depthwise_conv2d_2, reshape_8)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_7 = paddle._C_ops.multiply(parameter_29, add_11)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_12 = paddle._C_ops.add(multiply_7, parameter_30)

        # pd_op.hardswish: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16)
        hardswish_3 = paddle._C_ops.hardswish(add_12)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_8 = paddle._C_ops.multiply(parameter_31, hardswish_3)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_13 = paddle._C_ops.add(multiply_8, parameter_32)

        # pd_op.conv2d: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 48x48x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(add_13, parameter_33, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf16, 0x48xf16) <- (48xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_34, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1x48x1x1xf16)
        add_14 = paddle._C_ops.add(conv2d_3, reshape_10)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_9 = paddle._C_ops.multiply(parameter_35, add_14)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_15 = paddle._C_ops.add(multiply_9, parameter_36)

        # pd_op.hardswish: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16)
        hardswish_4 = paddle._C_ops.hardswish(add_15)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_10 = paddle._C_ops.multiply(parameter_37, hardswish_4)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_16 = paddle._C_ops.add(multiply_10, parameter_38)

        # pd_op.depthwise_conv2d: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 48x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(add_16, parameter_39, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf16, 0x48xf16) <- (48xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_40, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1x48x1x1xf16)
        add_17 = paddle._C_ops.add(depthwise_conv2d_3, reshape_12)

        # pd_op.multiply: (-1x48x-1x-1xf16) <- (1xf16, -1x48x-1x-1xf16)
        multiply_11 = paddle._C_ops.multiply(parameter_41, add_17)

        # pd_op.add: (-1x48x-1x-1xf16) <- (-1x48x-1x-1xf16, 1xf16)
        add_18 = paddle._C_ops.add(multiply_11, parameter_42)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x48x-1x-1xf16, 96x48x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(add_18, parameter_43, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_44, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1x96x1x1xf16)
        add_19 = paddle._C_ops.add(conv2d_4, reshape_14)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_12 = paddle._C_ops.multiply(parameter_45, add_19)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_20 = paddle._C_ops.add(multiply_12, parameter_46)

        # pd_op.hardswish: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16)
        hardswish_5 = paddle._C_ops.hardswish(add_20)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_13 = paddle._C_ops.multiply(parameter_47, hardswish_5)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_21 = paddle._C_ops.add(multiply_13, parameter_48)

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 96x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(add_21, parameter_49, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_50, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1x96x1x1xf16)
        add_22 = paddle._C_ops.add(depthwise_conv2d_4, reshape_16)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_14 = paddle._C_ops.multiply(parameter_51, add_22)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_23 = paddle._C_ops.add(multiply_14, parameter_52)

        # pd_op.hardswish: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16)
        hardswish_6 = paddle._C_ops.hardswish(add_23)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_15 = paddle._C_ops.multiply(parameter_53, hardswish_6)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_24 = paddle._C_ops.add(multiply_15, parameter_54)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 96x96x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(add_24, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_56, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1x96x1x1xf16)
        add_25 = paddle._C_ops.add(conv2d_5, reshape_18)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_16 = paddle._C_ops.multiply(parameter_57, add_25)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_26 = paddle._C_ops.add(multiply_16, parameter_58)

        # pd_op.hardswish: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16)
        hardswish_7 = paddle._C_ops.hardswish(add_26)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_17 = paddle._C_ops.multiply(parameter_59, hardswish_7)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_27 = paddle._C_ops.add(multiply_17, parameter_60)

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 96x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(add_27, parameter_61, [2, 2], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_62, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1x96x1x1xf16)
        add_28 = paddle._C_ops.add(depthwise_conv2d_5, reshape_20)

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (1xf16, -1x96x-1x-1xf16)
        multiply_18 = paddle._C_ops.multiply(parameter_63, add_28)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, 1xf16)
        add_29 = paddle._C_ops.add(multiply_18, parameter_64)

        # pd_op.conv2d: (-1x192x-1x-1xf16) <- (-1x96x-1x-1xf16, 192x96x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(add_29, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_66, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_30 = paddle._C_ops.add(conv2d_6, reshape_22)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_19 = paddle._C_ops.multiply(parameter_67, add_30)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_31 = paddle._C_ops.add(multiply_19, parameter_68)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_8 = paddle._C_ops.hardswish(add_31)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_20 = paddle._C_ops.multiply(parameter_69, hardswish_8)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_32 = paddle._C_ops.add(multiply_20, parameter_70)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x1x5x5xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(add_32, parameter_71, [1, 1], [2, 2], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_72, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_33 = paddle._C_ops.add(depthwise_conv2d_6, reshape_24)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_21 = paddle._C_ops.multiply(parameter_73, add_33)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_34 = paddle._C_ops.add(multiply_21, parameter_74)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_9 = paddle._C_ops.hardswish(add_34)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_22 = paddle._C_ops.multiply(parameter_75, hardswish_9)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_35 = paddle._C_ops.add(multiply_22, parameter_76)

        # pd_op.conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x192x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(add_35, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_78, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_36 = paddle._C_ops.add(conv2d_7, reshape_26)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_23 = paddle._C_ops.multiply(parameter_79, add_36)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_37 = paddle._C_ops.add(multiply_23, parameter_80)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_10 = paddle._C_ops.hardswish(add_37)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_24 = paddle._C_ops.multiply(parameter_81, hardswish_10)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_38 = paddle._C_ops.add(multiply_24, parameter_82)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x1x5x5xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(add_38, parameter_83, [1, 1], [2, 2], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_84, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_39 = paddle._C_ops.add(depthwise_conv2d_7, reshape_28)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_25 = paddle._C_ops.multiply(parameter_85, add_39)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_40 = paddle._C_ops.add(multiply_25, parameter_86)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_11 = paddle._C_ops.hardswish(add_40)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_26 = paddle._C_ops.multiply(parameter_87, hardswish_11)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_41 = paddle._C_ops.add(multiply_26, parameter_88)

        # pd_op.conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x192x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(add_41, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_90, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_42 = paddle._C_ops.add(conv2d_8, reshape_30)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_27 = paddle._C_ops.multiply(parameter_91, add_42)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_43 = paddle._C_ops.add(multiply_27, parameter_92)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_12 = paddle._C_ops.hardswish(add_43)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_28 = paddle._C_ops.multiply(parameter_93, hardswish_12)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_44 = paddle._C_ops.add(multiply_28, parameter_94)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x1x5x5xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add_44, parameter_95, [1, 1], [2, 2], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_96, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_45 = paddle._C_ops.add(depthwise_conv2d_8, reshape_32)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_29 = paddle._C_ops.multiply(parameter_97, add_45)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_46 = paddle._C_ops.add(multiply_29, parameter_98)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_13 = paddle._C_ops.hardswish(add_46)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_30 = paddle._C_ops.multiply(parameter_99, hardswish_13)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_47 = paddle._C_ops.add(multiply_30, parameter_100)

        # pd_op.conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x192x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(add_47, parameter_101, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_102, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_48 = paddle._C_ops.add(conv2d_9, reshape_34)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_31 = paddle._C_ops.multiply(parameter_103, add_48)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_49 = paddle._C_ops.add(multiply_31, parameter_104)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_14 = paddle._C_ops.hardswish(add_49)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_32 = paddle._C_ops.multiply(parameter_105, hardswish_14)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_50 = paddle._C_ops.add(multiply_32, parameter_106)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x1x5x5xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(add_50, parameter_107, [1, 1], [2, 2], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_108, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_51 = paddle._C_ops.add(depthwise_conv2d_9, reshape_36)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_33 = paddle._C_ops.multiply(parameter_109, add_51)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_52 = paddle._C_ops.add(multiply_33, parameter_110)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_15 = paddle._C_ops.hardswish(add_52)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_34 = paddle._C_ops.multiply(parameter_111, hardswish_15)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_53 = paddle._C_ops.add(multiply_34, parameter_112)

        # pd_op.conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x192x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(add_53, parameter_113, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_114, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_54 = paddle._C_ops.add(conv2d_10, reshape_38)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_35 = paddle._C_ops.multiply(parameter_115, add_54)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_55 = paddle._C_ops.add(multiply_35, parameter_116)

        # pd_op.hardswish: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16)
        hardswish_16 = paddle._C_ops.hardswish(add_55)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_36 = paddle._C_ops.multiply(parameter_117, hardswish_16)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_56 = paddle._C_ops.add(multiply_36, parameter_118)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 192x1x5x5xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(add_56, parameter_119, [2, 2], [2, 2], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_120, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1x192x1x1xf16)
        add_57 = paddle._C_ops.add(depthwise_conv2d_10, reshape_40)

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (1xf16, -1x192x-1x-1xf16)
        multiply_37 = paddle._C_ops.multiply(parameter_121, add_57)

        # pd_op.add: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, 1xf16)
        add_58 = paddle._C_ops.add(multiply_37, parameter_122)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf16) <- (-1x192x-1x-1xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(add_58, full_int_array_21, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf16) <- (-1x192x1x1xf16, 48x192x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_0, parameter_123, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf16, 0x48xf16) <- (48xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_124, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf16) <- (-1x48x1x1xf16, 1x48x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_11, reshape_42)

        # pd_op.relu_: (-1x48x1x1xf16) <- (-1x48x1x1xf16)
        relu__0 = paddle._C_ops.relu_(add__0)

        # pd_op.conv2d: (-1x192x1x1xf16) <- (-1x48x1x1xf16, 192x48x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__0, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_126, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf16) <- (-1x192x1x1xf16, 1x192x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_12, reshape_44)

        # pd_op.hardsigmoid: (-1x192x1x1xf16) <- (-1x192x1x1xf16)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add__1, float('0.166667'), float('0.5'))

        # pd_op.multiply: (-1x192x-1x-1xf16) <- (-1x192x-1x-1xf16, -1x192x1x1xf16)
        multiply_38 = paddle._C_ops.multiply(add_58, hardsigmoid_0)

        # pd_op.conv2d: (-1x384x-1x-1xf16) <- (-1x192x-1x-1xf16, 384x192x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(multiply_38, parameter_127, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_128, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_59 = paddle._C_ops.add(conv2d_13, reshape_46)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_39 = paddle._C_ops.multiply(parameter_129, add_59)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_60 = paddle._C_ops.add(multiply_39, parameter_130)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_17 = paddle._C_ops.hardswish(add_60)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_40 = paddle._C_ops.multiply(parameter_131, hardswish_17)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_61 = paddle._C_ops.add(multiply_40, parameter_132)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x1x5x5xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(add_61, parameter_133, [1, 1], [2, 2], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_134, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_62 = paddle._C_ops.add(depthwise_conv2d_11, reshape_48)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_41 = paddle._C_ops.multiply(parameter_135, add_62)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_63 = paddle._C_ops.add(multiply_41, parameter_136)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_18 = paddle._C_ops.hardswish(add_63)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_42 = paddle._C_ops.multiply(parameter_137, hardswish_18)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_64 = paddle._C_ops.add(multiply_42, parameter_138)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1, 1]

        # pd_op.pool2d: (-1x384x1x1xf16) <- (-1x384x-1x-1xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(add_64, full_int_array_26, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x96x1x1xf16) <- (-1x384x1x1xf16, 96x384x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(pool2d_1, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_140, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf16) <- (-1x96x1x1xf16, 1x96x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_14, reshape_50)

        # pd_op.relu_: (-1x96x1x1xf16) <- (-1x96x1x1xf16)
        relu__1 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x384x1x1xf16) <- (-1x96x1x1xf16, 384x96x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__1, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_142, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x1x1xf16) <- (-1x384x1x1xf16, 1x384x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_15, reshape_52)

        # pd_op.hardsigmoid: (-1x384x1x1xf16) <- (-1x384x1x1xf16)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add__3, float('0.166667'), float('0.5'))

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, -1x384x1x1xf16)
        multiply_43 = paddle._C_ops.multiply(add_64, hardsigmoid_1)

        # pd_op.conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x384x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(multiply_43, parameter_143, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_144, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_65 = paddle._C_ops.add(conv2d_16, reshape_54)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_44 = paddle._C_ops.multiply(parameter_145, add_65)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_66 = paddle._C_ops.add(multiply_44, parameter_146)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_19 = paddle._C_ops.hardswish(add_66)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_45 = paddle._C_ops.multiply(parameter_147, hardswish_19)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_67 = paddle._C_ops.add(multiply_45, parameter_148)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x1x5x5xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(add_67, parameter_149, [1, 1], [2, 2], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_150, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_68 = paddle._C_ops.add(depthwise_conv2d_12, reshape_56)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_46 = paddle._C_ops.multiply(parameter_151, add_68)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_69 = paddle._C_ops.add(multiply_46, parameter_152)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_20 = paddle._C_ops.hardswish(add_69)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_47 = paddle._C_ops.multiply(parameter_153, hardswish_20)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_70 = paddle._C_ops.add(multiply_47, parameter_154)

        # pd_op.conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x384x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(add_70, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_156, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_71 = paddle._C_ops.add(conv2d_17, reshape_58)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_48 = paddle._C_ops.multiply(parameter_157, add_71)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_72 = paddle._C_ops.add(multiply_48, parameter_158)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_21 = paddle._C_ops.hardswish(add_72)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_49 = paddle._C_ops.multiply(parameter_159, hardswish_21)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_73 = paddle._C_ops.add(multiply_49, parameter_160)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x1x5x5xf16)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add_73, parameter_161, [1, 1], [2, 2], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_162, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_74 = paddle._C_ops.add(depthwise_conv2d_13, reshape_60)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_50 = paddle._C_ops.multiply(parameter_163, add_74)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_75 = paddle._C_ops.add(multiply_50, parameter_164)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_22 = paddle._C_ops.hardswish(add_75)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_51 = paddle._C_ops.multiply(parameter_165, hardswish_22)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_76 = paddle._C_ops.add(multiply_51, parameter_166)

        # pd_op.conv2d: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 384x384x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(add_76, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_168, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1x384x1x1xf16)
        add_77 = paddle._C_ops.add(conv2d_18, reshape_62)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_52 = paddle._C_ops.multiply(parameter_169, add_77)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_78 = paddle._C_ops.add(multiply_52, parameter_170)

        # pd_op.hardswish: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16)
        hardswish_23 = paddle._C_ops.hardswish(add_78)

        # pd_op.multiply: (-1x384x-1x-1xf16) <- (1xf16, -1x384x-1x-1xf16)
        multiply_53 = paddle._C_ops.multiply(parameter_171, hardswish_23)

        # pd_op.add: (-1x384x-1x-1xf16) <- (-1x384x-1x-1xf16, 1xf16)
        add_79 = paddle._C_ops.add(multiply_53, parameter_172)

        # pd_op.conv2d: (-1x12x-1x-1xf16) <- (-1x48x-1x-1xf16, 12x48x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(add_16, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 12, 1, 1]

        # pd_op.reshape: (1x12x1x1xf16, 0x12xf16) <- (12xf16, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_174, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x12x-1x-1xf16) <- (-1x12x-1x-1xf16, 1x12x1x1xf16)
        add_80 = paddle._C_ops.add(conv2d_19, reshape_64)

        # pd_op.conv2d: (-1x18x-1x-1xf16) <- (-1x96x-1x-1xf16, 18x96x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(add_27, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 18, 1, 1]

        # pd_op.reshape: (1x18x1x1xf16, 0x18xf16) <- (18xf16, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_176, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x18x-1x-1xf16) <- (-1x18x-1x-1xf16, 1x18x1x1xf16)
        add_81 = paddle._C_ops.add(conv2d_20, reshape_66)

        # pd_op.conv2d: (-1x42x-1x-1xf16) <- (-1x192x-1x-1xf16, 42x192x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(add_56, parameter_177, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 42, 1, 1]

        # pd_op.reshape: (1x42x1x1xf16, 0x42xf16) <- (42xf16, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_178, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x42x-1x-1xf16) <- (-1x42x-1x-1xf16, 1x42x1x1xf16)
        add_82 = paddle._C_ops.add(conv2d_21, reshape_68)

        # pd_op.conv2d: (-1x360x-1x-1xf16) <- (-1x384x-1x-1xf16, 360x384x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(add_79, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 360, 1, 1]

        # pd_op.reshape: (1x360x1x1xf16, 0x360xf16) <- (360xf16, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_180, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x360x-1x-1xf16) <- (-1x360x-1x-1xf16, 1x360x1x1xf16)
        add_83 = paddle._C_ops.add(conv2d_22, reshape_70)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x360x-1x-1xf16, 96x360x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(add_83, parameter_181, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf16) <- (-1x96x-1x-1xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(conv2d_23, full_int_array_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x96x1x1xf16, 24x96x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(pool2d_2, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_24, reshape_72)

        # pd_op.relu_: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        relu__2 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x96x1x1xf16) <- (-1x24x1x1xf16, 96x24x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__2, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf16) <- (-1x96x1x1xf16, 1x96x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_25, reshape_74)

        # pd_op.hardsigmoid: (-1x96x1x1xf16) <- (-1x96x1x1xf16)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add__5, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x1x1xf16)
        multiply_54 = paddle._C_ops.multiply(conv2d_23, hardsigmoid_2)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_84 = paddle._C_ops.add(conv2d_23, multiply_54)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x42x-1x-1xf16, 96x42x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(add_82, parameter_186, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf16) <- (-1x96x-1x-1xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(conv2d_26, full_int_array_41, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x96x1x1xf16, 24x96x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(pool2d_3, parameter_187, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_188, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_27, reshape_76)

        # pd_op.relu_: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        relu__3 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x96x1x1xf16) <- (-1x24x1x1xf16, 96x24x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__3, parameter_189, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_190, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf16) <- (-1x96x1x1xf16, 1x96x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_28, reshape_78)

        # pd_op.hardsigmoid: (-1x96x1x1xf16) <- (-1x96x1x1xf16)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add__7, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x1x1xf16)
        multiply_55 = paddle._C_ops.multiply(conv2d_26, hardsigmoid_3)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_85 = paddle._C_ops.add(conv2d_26, multiply_55)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x18x-1x-1xf16, 96x18x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(add_81, parameter_191, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf16) <- (-1x96x-1x-1xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(conv2d_29, full_int_array_44, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x96x1x1xf16, 24x96x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(pool2d_4, parameter_192, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_193, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_30, reshape_80)

        # pd_op.relu_: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        relu__4 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x96x1x1xf16) <- (-1x24x1x1xf16, 96x24x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__4, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf16) <- (-1x96x1x1xf16, 1x96x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_31, reshape_82)

        # pd_op.hardsigmoid: (-1x96x1x1xf16) <- (-1x96x1x1xf16)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add__9, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x1x1xf16)
        multiply_56 = paddle._C_ops.multiply(conv2d_29, hardsigmoid_4)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_86 = paddle._C_ops.add(conv2d_29, multiply_56)

        # pd_op.conv2d: (-1x96x-1x-1xf16) <- (-1x12x-1x-1xf16, 96x12x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(add_80, parameter_196, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_47 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf16) <- (-1x96x-1x-1xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(conv2d_32, full_int_array_47, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x96x1x1xf16, 24x96x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(pool2d_5, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_198, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_33, reshape_84)

        # pd_op.relu_: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        relu__5 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x96x1x1xf16) <- (-1x24x1x1xf16, 96x24x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__5, parameter_199, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_200, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf16) <- (-1x96x1x1xf16, 1x96x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_34, reshape_86)

        # pd_op.hardsigmoid: (-1x96x1x1xf16) <- (-1x96x1x1xf16)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add__11, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x1x1xf16)
        multiply_57 = paddle._C_ops.multiply(conv2d_32, hardsigmoid_5)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_87 = paddle._C_ops.add(conv2d_32, multiply_57)

        # pd_op.nearest_interp: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(add_84, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_88 = paddle._C_ops.add(add_85, nearest_interp_0)

        # pd_op.nearest_interp: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(add_88, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_89 = paddle._C_ops.add(add_86, nearest_interp_1)

        # pd_op.nearest_interp: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(add_89, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf16) <- (-1x96x-1x-1xf16, -1x96x-1x-1xf16)
        add_90 = paddle._C_ops.add(add_87, nearest_interp_2)

        # pd_op.conv2d: (-1x24x-1x-1xf16) <- (-1x96x-1x-1xf16, 24x96x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(add_84, parameter_201, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [1, 1]

        # pd_op.pool2d: (-1x24x1x1xf16) <- (-1x24x-1x-1xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(conv2d_35, full_int_array_50, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf16) <- (-1x24x1x1xf16, 6x24x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(pool2d_6, parameter_202, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 6, 1, 1]

        # pd_op.reshape: (1x6x1x1xf16, 0x6xf16) <- (6xf16, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_203, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x6x1x1xf16) <- (-1x6x1x1xf16, 1x6x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_36, reshape_88)

        # pd_op.relu_: (-1x6x1x1xf16) <- (-1x6x1x1xf16)
        relu__6 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x6x1x1xf16, 24x6x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__6, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_205, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_37, reshape_90)

        # pd_op.hardsigmoid: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add__13, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x1x1xf16)
        multiply_58 = paddle._C_ops.multiply(conv2d_35, hardsigmoid_6)

        # pd_op.add: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x-1x-1xf16)
        add_91 = paddle._C_ops.add(conv2d_35, multiply_58)

        # pd_op.conv2d: (-1x24x-1x-1xf16) <- (-1x96x-1x-1xf16, 24x96x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(add_88, parameter_206, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_53 = [1, 1]

        # pd_op.pool2d: (-1x24x1x1xf16) <- (-1x24x-1x-1xf16, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(conv2d_38, full_int_array_53, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf16) <- (-1x24x1x1xf16, 6x24x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(pool2d_7, parameter_207, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 6, 1, 1]

        # pd_op.reshape: (1x6x1x1xf16, 0x6xf16) <- (6xf16, 4xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_208, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x6x1x1xf16) <- (-1x6x1x1xf16, 1x6x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_39, reshape_92)

        # pd_op.relu_: (-1x6x1x1xf16) <- (-1x6x1x1xf16)
        relu__7 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x6x1x1xf16, 24x6x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__7, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_210, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_40, reshape_94)

        # pd_op.hardsigmoid: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add__15, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x1x1xf16)
        multiply_59 = paddle._C_ops.multiply(conv2d_38, hardsigmoid_7)

        # pd_op.add: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x-1x-1xf16)
        add_92 = paddle._C_ops.add(conv2d_38, multiply_59)

        # pd_op.conv2d: (-1x24x-1x-1xf16) <- (-1x96x-1x-1xf16, 24x96x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(add_89, parameter_211, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_56 = [1, 1]

        # pd_op.pool2d: (-1x24x1x1xf16) <- (-1x24x-1x-1xf16, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(conv2d_41, full_int_array_56, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf16) <- (-1x24x1x1xf16, 6x24x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(pool2d_8, parameter_212, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 6, 1, 1]

        # pd_op.reshape: (1x6x1x1xf16, 0x6xf16) <- (6xf16, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_213, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x6x1x1xf16) <- (-1x6x1x1xf16, 1x6x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_42, reshape_96)

        # pd_op.relu_: (-1x6x1x1xf16) <- (-1x6x1x1xf16)
        relu__8 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x6x1x1xf16, 24x6x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__8, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_43, reshape_98)

        # pd_op.hardsigmoid: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        hardsigmoid_8 = paddle._C_ops.hardsigmoid(add__17, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x1x1xf16)
        multiply_60 = paddle._C_ops.multiply(conv2d_41, hardsigmoid_8)

        # pd_op.add: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x-1x-1xf16)
        add_93 = paddle._C_ops.add(conv2d_41, multiply_60)

        # pd_op.conv2d: (-1x24x-1x-1xf16) <- (-1x96x-1x-1xf16, 24x96x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(add_90, parameter_216, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_59 = [1, 1]

        # pd_op.pool2d: (-1x24x1x1xf16) <- (-1x24x-1x-1xf16, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(conv2d_44, full_int_array_59, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf16) <- (-1x24x1x1xf16, 6x24x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_9, parameter_217, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 6, 1, 1]

        # pd_op.reshape: (1x6x1x1xf16, 0x6xf16) <- (6xf16, 4xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_218, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x6x1x1xf16) <- (-1x6x1x1xf16, 1x6x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_45, reshape_100)

        # pd_op.relu_: (-1x6x1x1xf16) <- (-1x6x1x1xf16)
        relu__9 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x24x1x1xf16) <- (-1x6x1x1xf16, 24x6x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__9, parameter_219, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_220, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf16) <- (-1x24x1x1xf16, 1x24x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_46, reshape_102)

        # pd_op.hardsigmoid: (-1x24x1x1xf16) <- (-1x24x1x1xf16)
        hardsigmoid_9 = paddle._C_ops.hardsigmoid(add__19, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x1x1xf16)
        multiply_61 = paddle._C_ops.multiply(conv2d_44, hardsigmoid_9)

        # pd_op.add: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, -1x24x-1x-1xf16)
        add_94 = paddle._C_ops.add(conv2d_44, multiply_61)

        # pd_op.nearest_interp: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(add_91, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'nearest', False, 1)

        # pd_op.nearest_interp: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(add_92, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 1)

        # pd_op.nearest_interp: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, None, None, None)
        nearest_interp_5 = paddle._C_ops.nearest_interp(add_93, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # builtin.combine: ([-1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16]) <- (-1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16)
        combine_0 = [nearest_interp_3, nearest_interp_4, nearest_interp_5, add_94]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x-1x-1xf16) <- ([-1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16, -1x24x-1x-1xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x24x-1x-1xf16) <- (-1x96x-1x-1xf16, 24x96x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(concat_0, parameter_221, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x-1x-1xf16, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x-1x-1xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16)
        relu_0 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.cast: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf16)
        cast_1 = paddle._C_ops.cast(relu_0, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_62 = []

        # pd_op.conv2d_transpose: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 24x24x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(cast_1, parameter_226, [2, 2], [0, 0], [], full_int_array_62, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf16, 0x24xf16) <- (24xf16, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_227, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(conv2d_transpose_0, paddle.float16)

        # pd_op.add: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, 1x24x1x1xf16)
        add_95 = paddle._C_ops.add(cast_2, reshape_104)

        # pd_op.batch_norm_: (-1x24x-1x-1xf16, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x-1x-1xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_95, parameter_228, parameter_229, parameter_230, parameter_231, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16)
        relu_1 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.cast: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf16)
        cast_3 = paddle._C_ops.cast(relu_1, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_64 = []

        # pd_op.conv2d_transpose: (-1x1x-1x-1xf32) <- (-1x24x-1x-1xf32, 24x1x2x2xf32, 0xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(cast_3, parameter_232, [2, 2], [0, 0], [], full_int_array_64, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_233, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf32)
        cast_4 = paddle._C_ops.cast(conv2d_transpose_1, paddle.float16)

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_96 = paddle._C_ops.add(cast_4, reshape_106)

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_0 = paddle._C_ops.sigmoid(add_96)

        # pd_op.nearest_interp: (-1x24x-1x-1xf16) <- (-1x24x-1x-1xf16, None, None, None)
        nearest_interp_6 = paddle._C_ops.nearest_interp(relu_1, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # builtin.combine: ([-1x1x-1x-1xf16, -1x24x-1x-1xf16]) <- (-1x1x-1x-1xf16, -1x24x-1x-1xf16)
        combine_1 = [sigmoid_0, nearest_interp_6]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x25x-1x-1xf16) <- ([-1x1x-1x-1xf16, -1x24x-1x-1xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.conv2d: (-1x12x-1x-1xf16) <- (-1x25x-1x-1xf16, 12x25x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(concat_1, parameter_234, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x12x-1x-1xf16, 12xf32, 12xf32, 12xf32, 12xf32, None) <- (-1x12x-1x-1xf16, 12xf32, 12xf32, 12xf32, 12xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x12x-1x-1xf16) <- (-1x12x-1x-1xf16)
        relu_2 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x12x-1x-1xf16, 1x12x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu_2, parameter_239, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_240, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_97 = paddle._C_ops.add(conv2d_49, reshape_108)

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_1 = paddle._C_ops.sigmoid(add_97)

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, -1x1x-1x-1xf16)
        add_98 = paddle._C_ops.add(sigmoid_0, sigmoid_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1xf32)
        scale_0 = paddle._C_ops.scale(add_98, full_2, float('0'), True)

        # pd_op.cast: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf16)
        cast_5 = paddle._C_ops.cast(sigmoid_1, paddle.float32)

        # pd_op.cast: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf16)
        cast_6 = paddle._C_ops.cast(scale_0, paddle.float32)
        return cast_5, cast_6



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_240, feed_0):
        return self.builtin_module_2105_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_240, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2105_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([16, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([48, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([48], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([48, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([48], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([48, 48, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([48], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([48, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([48], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([96, 48, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([96, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([96, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([192, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([192, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([192, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([192, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([192, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([48, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([48], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([192, 48, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([384, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([384, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([96, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([384, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([384, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([384, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([12, 48, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([12], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([18, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([18], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([42, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([42], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([360, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([360], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([96, 360, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([24, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([96, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([96, 42, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([24, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([96, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([96, 18, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([24, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([96, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([96, 12, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([24, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([96, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([6, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([6], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([24, 6, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([6, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([6], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([24, 6, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([6, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([6], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([24, 6, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([6, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([6], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([24, 6, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([24, 24, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([24], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([24, 1, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([12, 25, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1, 12, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 960, 960], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[16], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[32, 16, 1, 1], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[48, 32, 1, 1], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[48], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[48], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[48], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[48], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[96, 48, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[192, 1, 5, 5], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[192, 1, 5, 5], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[192, 1, 5, 5], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[192, 1, 5, 5], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[192, 1, 5, 5], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[48], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[192, 48, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[384, 192, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[384, 1, 5, 5], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[96, 384, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[384, 96, 1, 1], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[384, 1, 5, 5], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[384, 1, 5, 5], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[12], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[18, 96, 1, 1], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[18], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[42, 192, 1, 1], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[42], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[360, 384, 1, 1], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[360], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[96, 360, 1, 1], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[96, 42, 1, 1], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[96, 18, 1, 1], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[96, 12, 1, 1], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[6], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[6], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[6], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[6], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[24, 24, 2, 2], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[24], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[24, 1, 2, 2], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[12, 25, 3, 3], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[1, 12, 1, 1], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
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