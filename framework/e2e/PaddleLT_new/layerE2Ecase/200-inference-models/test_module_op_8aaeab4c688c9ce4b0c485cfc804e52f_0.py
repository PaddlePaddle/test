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
    return [341][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_598_0_0(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_136, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_247, parameter_244, parameter_246, parameter_245, parameter_248, parameter_249, feed_0):

        # pd_op.conv2d: (1x64x256x256xf32) <- (1x3x512x512xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x256x256xf32) <- (1x64x256x256xf32, 1x64x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (1x64x256x256xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x256xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x256x256xf32) <- (1x64x256x256xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (1x64x128x128xf32) <- (1x64x256x256xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_6, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x128x128xf32) <- (1x64x128x128xf32, 1x64x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (1x64x128x128xf32) <- (1x64x128x128xf32, 64x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_13, parameter_14, parameter_15, parameter_16, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (1x64x128x128xf32) <- (1x64x128x128xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_17, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_18, parameter_19, parameter_20, parameter_21, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x128x128xf32) <- (1x64x128x128xf32, 1x64x128x128xf32)
        add__2 = paddle._C_ops.add_(batch_norm__18, relu__1)

        # pd_op.relu_: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu__3 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (1x64x128x128xf32) <- (1x64x128x128xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_23, parameter_24, parameter_25, parameter_26, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.conv2d: (1x64x128x128xf32) <- (1x64x128x128xf32, 64x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__4, parameter_27, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_28, parameter_29, parameter_30, parameter_31, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x128x128xf32) <- (1x64x128x128xf32, 1x64x128x128xf32)
        add__3 = paddle._C_ops.add_(batch_norm__30, relu__3)

        # pd_op.relu_: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu__5 = paddle._C_ops.relu_(add__3)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x64x128x128xf32, 128x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_32, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__6, parameter_37, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x64x128x128xf32, 128x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__5, parameter_42, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__4 = paddle._C_ops.add_(batch_norm__42, batch_norm__48)

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__7 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__8, parameter_52, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__5 = paddle._C_ops.add_(batch_norm__60, relu__7)

        # pd_op.relu: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu_0 = paddle._C_ops.relu(add__5)

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x128x64x64xf32, 256x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_0, parameter_57, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x256x32x32xf32, 256x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__9, parameter_62, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_63, parameter_64, parameter_65, parameter_66, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x128x64x64xf32, 256x128x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu_0, parameter_67, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x32x32xf32) <- (1x256x32x32xf32, 1x256x32x32xf32)
        add__6 = paddle._C_ops.add_(batch_norm__72, batch_norm__78)

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__10 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x256x32x32xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__10, parameter_72, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__11 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x256x32x32xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__11, parameter_77, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x32x32xf32) <- (1x256x32x32xf32, 1x256x32x32xf32)
        add__7 = paddle._C_ops.add_(batch_norm__90, relu__10)

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__7, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x256x32x32xf32, 128x256x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__12, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (1x128x64x64xf32)
        shape_0 = paddle._C_ops.shape(add__5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_2, full_int_array_3, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__0 = paddle._C_ops.cast_(slice_0, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x32x32xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(conv2d_16, cast__0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x-1x-1xf32)
        add__8 = paddle._C_ops.add_(add__5, bilinear_interp_0)

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__13 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__13, parameter_87, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__14, parameter_92, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_93, parameter_94, parameter_95, parameter_96, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__9 = paddle._C_ops.add_(batch_norm__108, relu__13)

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__15 = paddle._C_ops.relu_(add__9)

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__16 = paddle._C_ops.relu_(relu__15)

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__17 = paddle._C_ops.relu_(add__7)

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(relu__17, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x256x32x32xf32, 512x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__114, parameter_101, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(relu__17, parameter_102, parameter_103, parameter_104, parameter_105, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x256x32x32xf32, 512x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(batch_norm__120, parameter_106, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (1x512x16x16xf32)
        shape_1 = paddle._C_ops.shape(conv2d_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [4]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [0, 8, 64, -1]

        # pd_op.reshape_: (1x8x64x256xf32, 0x1x512x16x16xf32) <- (1x512x16x16xf32, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_20, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (1x8x64x256xf32) <- (1x8x64x256xf32)
        softmax__0 = paddle._C_ops.softmax_(reshape__0, 3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.sum: (1x8x1x256xf32) <- (1x8x64x256xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(softmax__0, full_int_array_9, None, True)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x8x1x256xf32) <- (1x8x1x256xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(sum_0, full_0, float('1e-06'), True)

        # pd_op.divide_: (1x8x64x256xf32) <- (1x8x64x256xf32, 1x8x1x256xf32)
        divide__0 = paddle._C_ops.divide_(softmax__0, scale__0)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, xi32, xi32]) <- (1xi32, 1xi32, xi32, xi32)
        combine_0 = [full_1, full_2, slice_1, slice_2]

        # pd_op.reshape_: (1x512x-1x-1xf32, 0x1x8x64x256xf32) <- (1x8x64x256xf32, [1xi32, 1xi32, xi32, xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(divide__0, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(reshape__2, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x-1x-1xf32)
        add__10 = paddle._C_ops.add_(conv2d_19, conv2d_21)

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__10, parameter_108, parameter_109, parameter_110, parameter_111, True, float('0.1'), float('1e-06'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x512x16x16xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(batch_norm__126, parameter_112, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_22, reshape_4)

        # pd_op.gelu: (1x512x16x16xf32) <- (1x512x16x16xf32)
        gelu_0 = paddle._C_ops.gelu(add__11, False)

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x512x16x16xf32, 512x512x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(gelu_0, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_23, reshape_6)

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x16x16xf32)
        add__13 = paddle._C_ops.add_(add__10, add__12)

        # pd_op.shape: (4xi32) <- (1x128x64x64xf32)
        shape_2 = paddle._C_ops.shape(relu__16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_2, [0], full_int_array_12, full_int_array_13, [1], [])

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__13, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x16x16xf32) <- (1x512x16x16xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x512x16x16xf32, 128x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__18, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__1 = paddle._C_ops.cast_(slice_3, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x16x16xf32, 2xi32, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(conv2d_24, cast__1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x-1x-1xf32)
        add__14 = paddle._C_ops.add_(relu__16, bilinear_interp_1)

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__13, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.max_pool2d_with_index: (1x512x12x12xf32, 1x512x12x12xi32) <- (1x512x16x16xf32)
        max_pool2d_with_index_0, max_pool2d_with_index_1 = (lambda x, f: f(x))(paddle._C_ops.max_pool2d_with_index(batch_norm__138, [12, 12], [1, 1], [0, 0], False, True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x12x12xf32) <- (1x512x12x12xf32, 256x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(max_pool2d_with_index_0, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x128x12x12xf32, 1x128x12x12xf32]) <- (1x256x12x12xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(conv2d_25, 2, full_3)

        # builtin.slice: (1x128x12x12xf32) <- ([1x128x12x12xf32, 1x128x12x12xf32])
        slice_4 = split_with_num_0[0]

        # pd_op.transpose: (1x12x12x128xf32) <- (1x128x12x12xf32)
        transpose_0 = paddle._C_ops.transpose(slice_4, [0, 2, 3, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [-1, 128, 1, 1]

        # pd_op.reshape_: (144x128x1x1xf32, 0x1x12x12x128xf32) <- (1x12x12x128xf32, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.slice: (1x128x12x12xf32) <- ([1x128x12x12xf32, 1x128x12x12xf32])
        slice_5 = split_with_num_0[1]

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [-1, 144, 1, 1]

        # pd_op.reshape_: (128x144x1x1xf32, 0x1x128x12x12xf32) <- (1x128x12x12xf32, 4xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_5, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__14, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, -1, 0, 0]

        # pd_op.reshape_: (1x128x64x64xf32, 0x1x128x64x64xf32) <- (1x128x64x64xf32, 4xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__144, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x144x64x64xf32) <- (1x128x64x64xf32, 144x128x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(reshape__8, reshape__4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [-1, 144, 0, 0]

        # pd_op.reshape_: (1x144x64x64xf32, 0x1x144x64x64xf32) <- (1x144x64x64xf32, 4xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_26, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('0.0833333'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x144x64x64xf32) <- (1x144x64x64xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(reshape__10, full_4, float('0'), True)

        # pd_op.softmax_: (1x144x64x64xf32) <- (1x144x64x64xf32)
        softmax__1 = paddle._C_ops.softmax_(scale__1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, -1, 0, 0]

        # pd_op.reshape_: (1x144x64x64xf32, 0x1x144x64x64xf32) <- (1x144x64x64xf32, 4xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__1, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x144x64x64xf32, 128x144x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(reshape__12, reshape__6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [-1, 128, 0, 0]

        # pd_op.reshape_: (1x128x64x64xf32, 0x1x128x64x64xf32) <- (1x128x64x64xf32, 4xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_27, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__15 = paddle._C_ops.add_(add__14, reshape__14)

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__15, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.1'), float('1e-06'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(batch_norm__150, parameter_134, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_135, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_28, reshape_8)

        # pd_op.gelu: (1x128x64x64xf32) <- (1x128x64x64xf32)
        gelu_1 = paddle._C_ops.gelu(add__16, False)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(gelu_1, parameter_136, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_137, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_29, reshape_10)

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__18 = paddle._C_ops.add_(add__15, add__17)

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__18, parameter_138, parameter_139, parameter_140, parameter_141, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__19 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x128x64x64xf32, 256x128x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__19, parameter_142, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_143, parameter_144, parameter_145, parameter_146, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu__20 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x256x32x32xf32, 512x256x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__20, parameter_147, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x16x16xf32)
        add__19 = paddle._C_ops.add_(add__13, conv2d_31)

        # pd_op.relu_: (1x128x64x64xf32) <- (1x128x64x64xf32)
        relu__21 = paddle._C_ops.relu_(add__18)

        # pd_op.relu_: (1x512x16x16xf32) <- (1x512x16x16xf32)
        relu__22 = paddle._C_ops.relu_(add__19)

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(relu__22, parameter_148, parameter_149, parameter_150, parameter_151, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x512x16x16xf32, 512x512x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(batch_norm__168, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (1x512x16x16xf32)
        shape_3 = paddle._C_ops.shape(conv2d_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [3]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [4]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_3, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [0, 8, 64, -1]

        # pd_op.reshape_: (1x8x64x256xf32, 0x1x512x16x16xf32) <- (1x512x16x16xf32, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_32, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (1x8x64x256xf32) <- (1x8x64x256xf32)
        softmax__2 = paddle._C_ops.softmax_(reshape__16, 3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [2]

        # pd_op.sum: (1x8x1x256xf32) <- (1x8x64x256xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(softmax__2, full_int_array_27, None, True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x8x1x256xf32) <- (1x8x1x256xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(sum_1, full_5, float('1e-06'), True)

        # pd_op.divide_: (1x8x64x256xf32) <- (1x8x64x256xf32, 1x8x1x256xf32)
        divide__1 = paddle._C_ops.divide_(softmax__2, scale__2)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, xi32, xi32]) <- (1xi32, 1xi32, xi32, xi32)
        combine_1 = [full_6, full_7, slice_6, slice_7]

        # pd_op.reshape_: (1x512x-1x-1xf32, 0x1x8x64x256xf32) <- (1x8x64x256xf32, [1xi32, 1xi32, xi32, xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(divide__1, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x512x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(reshape__18, parameter_153, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x-1x-1xf32)
        add__20 = paddle._C_ops.add_(relu__22, conv2d_33)

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__20, parameter_154, parameter_155, parameter_156, parameter_157, True, float('0.1'), float('1e-06'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x512x16x16xf32, 512x512x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(batch_norm__174, parameter_158, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_159, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_34, reshape_12)

        # pd_op.gelu: (1x512x16x16xf32) <- (1x512x16x16xf32)
        gelu_2 = paddle._C_ops.gelu(add__21, False)

        # pd_op.conv2d: (1x512x16x16xf32) <- (1x512x16x16xf32, 512x512x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(gelu_2, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_35, reshape_14)

        # pd_op.add_: (1x512x16x16xf32) <- (1x512x16x16xf32, 1x512x16x16xf32)
        add__23 = paddle._C_ops.add_(add__20, add__22)

        # pd_op.shape: (4xi32) <- (1x128x64x64xf32)
        shape_4 = paddle._C_ops.shape(relu__21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_4, [0], full_int_array_30, full_int_array_31, [1], [])

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_162, parameter_163, parameter_164, parameter_165, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x16x16xf32) <- (1x512x16x16xf32)
        relu__23 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x512x16x16xf32, 128x512x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__23, parameter_166, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__2 = paddle._C_ops.cast_(slice_8, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x16x16xf32, 2xi32, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(conv2d_36, cast__2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x-1x-1xf32)
        add__24 = paddle._C_ops.add_(relu__21, bilinear_interp_2)

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_167, parameter_168, parameter_169, parameter_170, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.max_pool2d_with_index: (1x512x12x12xf32, 1x512x12x12xi32) <- (1x512x16x16xf32)
        max_pool2d_with_index_2, max_pool2d_with_index_3 = (lambda x, f: f(x))(paddle._C_ops.max_pool2d_with_index(batch_norm__186, [12, 12], [1, 1], [0, 0], False, True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x12x12xf32) <- (1x512x12x12xf32, 256x512x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(max_pool2d_with_index_2, parameter_171, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x128x12x12xf32, 1x128x12x12xf32]) <- (1x256x12x12xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(conv2d_37, 2, full_8)

        # builtin.slice: (1x128x12x12xf32) <- ([1x128x12x12xf32, 1x128x12x12xf32])
        slice_9 = split_with_num_1[0]

        # pd_op.transpose: (1x12x12x128xf32) <- (1x128x12x12xf32)
        transpose_1 = paddle._C_ops.transpose(slice_9, [0, 2, 3, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [-1, 128, 1, 1]

        # pd_op.reshape_: (144x128x1x1xf32, 0x1x12x12x128xf32) <- (1x12x12x128xf32, 4xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.slice: (1x128x12x12xf32) <- ([1x128x12x12xf32, 1x128x12x12xf32])
        slice_10 = split_with_num_1[1]

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [-1, 144, 1, 1]

        # pd_op.reshape_: (128x144x1x1xf32, 0x1x128x12x12xf32) <- (1x128x12x12xf32, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_10, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__24, parameter_172, parameter_173, parameter_174, parameter_175, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, -1, 0, 0]

        # pd_op.reshape_: (1x128x64x64xf32, 0x1x128x64x64xf32) <- (1x128x64x64xf32, 4xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__192, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x144x64x64xf32) <- (1x128x64x64xf32, 144x128x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(reshape__24, reshape__20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [-1, 144, 0, 0]

        # pd_op.reshape_: (1x144x64x64xf32, 0x1x144x64x64xf32) <- (1x144x64x64xf32, 4xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_38, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('0.0833333'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x144x64x64xf32) <- (1x144x64x64xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(reshape__26, full_9, float('0'), True)

        # pd_op.softmax_: (1x144x64x64xf32) <- (1x144x64x64xf32)
        softmax__3 = paddle._C_ops.softmax_(scale__3, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, -1, 0, 0]

        # pd_op.reshape_: (1x144x64x64xf32, 0x1x144x64x64xf32) <- (1x144x64x64xf32, 4xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__3, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x144x64x64xf32, 128x144x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(reshape__28, reshape__22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [-1, 128, 0, 0]

        # pd_op.reshape_: (1x128x64x64xf32, 0x1x128x64x64xf32) <- (1x128x64x64xf32, 4xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_39, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__25 = paddle._C_ops.add_(add__24, reshape__30)

        # pd_op.batch_norm_: (1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__25, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.1'), float('1e-06'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(batch_norm__198, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_40, reshape_16)

        # pd_op.gelu: (1x128x64x64xf32) <- (1x128x64x64xf32)
        gelu_3 = paddle._C_ops.gelu(add__26, False)

        # pd_op.conv2d: (1x128x64x64xf32) <- (1x128x64x64xf32, 128x128x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(gelu_3, parameter_182, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_41, reshape_18)

        # pd_op.add_: (1x128x64x64xf32) <- (1x128x64x64xf32, 1x128x64x64xf32)
        add__28 = paddle._C_ops.add_(add__25, add__27)

        # pd_op.shape: (4xi32) <- (1x512x16x16xf32)
        shape_5 = paddle._C_ops.shape(add__23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_5, [0], full_int_array_40, full_int_array_41, [1], [])

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x16x16xf32) <- (1x512x16x16xf32)
        relu__24 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x512x16x16xf32, 128x512x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu__24, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [5, 5]

        # pd_op.pool2d: (1x512x8x8xf32) <- (1x512x16x16xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(add__23, full_int_array_42, [2, 2], [2, 2], False, False, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_0, parameter_189, parameter_190, parameter_191, parameter_192, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x8x8xf32) <- (1x512x8x8xf32)
        relu__25 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (1x128x8x8xf32) <- (1x512x8x8xf32, 128x512x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__25, parameter_193, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_0 = paddle._C_ops.cast(slice_11, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x8x8xf32, 2xi32, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(conv2d_43, cast_0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (1x128x16x16xf32) <- (1x128x-1x-1xf32, 1x128x16x16xf32)
        add_0 = paddle._C_ops.add(bilinear_interp_3, conv2d_42)

        # pd_op.batch_norm_: (1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_0, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x16x16xf32) <- (1x128x16x16xf32)
        relu__26 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x128x16x16xf32, 128x128x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__26, parameter_198, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [9, 9]

        # pd_op.pool2d: (1x512x4x4xf32) <- (1x512x16x16xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(add__23, full_int_array_43, [4, 4], [4, 4], False, False, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.batch_norm_: (1x512x4x4xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x4x4xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_1, parameter_199, parameter_200, parameter_201, parameter_202, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x4x4xf32) <- (1x512x4x4xf32)
        relu__27 = paddle._C_ops.relu_(batch_norm__222)

        # pd_op.conv2d: (1x128x4x4xf32) <- (1x512x4x4xf32, 128x512x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu__27, parameter_203, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_1 = paddle._C_ops.cast(slice_11, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x4x4xf32, 2xi32, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(conv2d_45, cast_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (1x128x16x16xf32) <- (1x128x-1x-1xf32, 1x128x16x16xf32)
        add_1 = paddle._C_ops.add(bilinear_interp_4, conv2d_44)

        # pd_op.batch_norm_: (1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_1, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x16x16xf32) <- (1x128x16x16xf32)
        relu__28 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x128x16x16xf32, 128x128x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu__28, parameter_208, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [17, 17]

        # pd_op.pool2d: (1x512x2x2xf32) <- (1x512x16x16xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(add__23, full_int_array_44, [8, 8], [8, 8], False, False, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.batch_norm_: (1x512x2x2xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x2x2xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_2, parameter_209, parameter_210, parameter_211, parameter_212, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x2x2xf32) <- (1x512x2x2xf32)
        relu__29 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.conv2d: (1x128x2x2xf32) <- (1x512x2x2xf32, 128x512x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__29, parameter_213, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_2 = paddle._C_ops.cast(slice_11, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x2x2xf32, 2xi32, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(conv2d_47, cast_2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (1x128x16x16xf32) <- (1x128x-1x-1xf32, 1x128x16x16xf32)
        add_2 = paddle._C_ops.add(bilinear_interp_5, conv2d_46)

        # pd_op.batch_norm_: (1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_2, parameter_214, parameter_215, parameter_216, parameter_217, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x16x16xf32) <- (1x128x16x16xf32)
        relu__30 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x128x16x16xf32, 128x128x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__30, parameter_218, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_45 = [1, 1]

        # pd_op.pool2d: (1x512x1x1xf32) <- (1x512x16x16xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(add__23, full_int_array_45, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.batch_norm_: (1x512x1x1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x1x1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_3, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x1x1xf32) <- (1x512x1x1xf32)
        relu__31 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.conv2d: (1x128x1x1xf32) <- (1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__31, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__3 = paddle._C_ops.cast_(slice_11, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x1x1xf32, 2xi32, None, None)
        bilinear_interp_6 = paddle._C_ops.bilinear_interp(conv2d_49, cast__3, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (1x128x16x16xf32) <- (1x128x-1x-1xf32, 1x128x16x16xf32)
        add_3 = paddle._C_ops.add(bilinear_interp_6, conv2d_48)

        # pd_op.batch_norm_: (1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_3, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x128x16x16xf32) <- (1x128x16x16xf32)
        relu__32 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x128x16x16xf32, 128x128x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu__32, parameter_228, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # builtin.combine: ([1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32]) <- (1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32)
        combine_2 = [conv2d_42, conv2d_44, conv2d_46, conv2d_48, conv2d_50]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x640x16x16xf32) <- ([1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32, 1x128x16x16xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_10)

        # pd_op.batch_norm_: (1x640x16x16xf32, 640xf32, 640xf32, xf32, xf32, None) <- (1x640x16x16xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_0, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x640x16x16xf32) <- (1x640x16x16xf32)
        relu__33 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x640x16x16xf32, 128x640x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__33, parameter_233, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_234, parameter_235, parameter_236, parameter_237, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x512x16x16xf32) <- (1x512x16x16xf32)
        relu__34 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (1x128x16x16xf32) <- (1x512x16x16xf32, 128x512x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__34, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (1x128x16x16xf32) <- (1x128x16x16xf32, 1x128x16x16xf32)
        add__29 = paddle._C_ops.add_(conv2d_51, conv2d_52)

        # pd_op.shape: (4xi32) <- (1x128x64x64xf32)
        shape_6 = paddle._C_ops.shape(add__28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_6, [0], full_int_array_46, full_int_array_47, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__4 = paddle._C_ops.cast_(slice_12, paddle.int32)

        # pd_op.bilinear_interp: (1x128x-1x-1xf32) <- (1x128x16x16xf32, 2xi32, None, None)
        bilinear_interp_7 = paddle._C_ops.bilinear_interp(add__29, cast__4, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # builtin.combine: ([1x128x64x64xf32, 1x128x-1x-1xf32]) <- (1x128x64x64xf32, 1x128x-1x-1xf32)
        combine_3 = [add__28, bilinear_interp_7]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x256x64x64xf32) <- ([1x128x64x64xf32, 1x128x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_11)

        # pd_op.batch_norm_: (1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_1, parameter_239, parameter_240, parameter_241, parameter_242, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x64x64xf32) <- (1x256x64x64xf32)
        relu__35 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.conv2d: (1x256x64x64xf32) <- (1x256x64x64xf32, 256x256x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__35, parameter_243, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_244, parameter_245, parameter_246, parameter_247, True, float('0.1'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x256x64x64xf32) <- (1x256x64x64xf32)
        relu__36 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (1x19x64x64xf32) <- (1x256x64x64xf32, 19x256x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__36, parameter_248, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 19, 1, 1]

        # pd_op.reshape: (1x19x1x1xf32, 0x19xf32) <- (19xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_249, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x19x64x64xf32) <- (1x19x64x64xf32, 1x19x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_54, reshape_20)

        # pd_op.shape: (4xi32) <- (1x3x512x512xf32)
        shape_7 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], full_int_array_49, full_int_array_50, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__5 = paddle._C_ops.cast_(slice_13, paddle.int32)

        # pd_op.bilinear_interp: (1x19x-1x-1xf32) <- (1x19x64x64xf32, 2xi32, None, None)
        bilinear_interp_8 = paddle._C_ops.bilinear_interp(add__30, cast__5, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.full: (1xi64) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (1x-1x-1xi32) <- (1x19x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(bilinear_interp_8, full_12, False, False, paddle.int32)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1x-1x-1xi32) <- (1x-1x-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(argmax_0, full_13, float('0'), True)
        return scale_0



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

    def forward(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_136, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_247, parameter_244, parameter_246, parameter_245, parameter_248, parameter_249, feed_0):
        return self.builtin_module_598_0_0(parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_114, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_135, parameter_136, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_170, parameter_167, parameter_169, parameter_168, parameter_171, parameter_175, parameter_172, parameter_174, parameter_173, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_247, parameter_244, parameter_246, parameter_245, parameter_248, parameter_249, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_598_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
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
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_127
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
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_113
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
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_127
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
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128, 640, 1, 1], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[19, 256, 1, 1], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[19], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 3, 512, 512], dtype='float32'),
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