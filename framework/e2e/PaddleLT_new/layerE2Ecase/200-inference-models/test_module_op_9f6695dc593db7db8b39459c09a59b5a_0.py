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
    return [939][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1275_0_0(self, parameter_0, parameter_1, parameter_2, parameter_6, parameter_3, parameter_5, parameter_4, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_211, parameter_208, parameter_210, parameter_209, parameter_212, parameter_216, parameter_213, parameter_215, parameter_214, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_226, parameter_223, parameter_225, parameter_224, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_288, parameter_290, parameter_289, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_306, parameter_305, parameter_307, parameter_308, parameter_310, parameter_309, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, feed_1, feed_0):

        # pd_op.cast: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.multiply: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf16, 1x3x1x1xf16)
        multiply_0 = paddle._C_ops.multiply(cast_0, parameter_0)

        # pd_op.add: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf16, 1x3x1x1xf16)
        add_0 = paddle._C_ops.add(multiply_0, parameter_1)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x3x-1x-1xf16, 64x3x7x7xf16)
        conv2d_0 = paddle._C_ops.conv2d(add_0, parameter_2, [2, 2], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_3, parameter_4, parameter_5, parameter_6, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16, 64x64x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_7, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_8, parameter_9, parameter_10, parameter_11, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16, 64x64x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_13, parameter_14, parameter_15, parameter_16, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x64x-1x-1xf16, 256x64x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu_2, parameter_17, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_18, parameter_19, parameter_20, parameter_21, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x64x-1x-1xf16, 256x64x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_0, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_23, parameter_24, parameter_25, parameter_26, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        add_1 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_3 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x256x-1x-1xf16, 64x256x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu_3, parameter_27, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_28, parameter_29, parameter_30, parameter_31, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_4 = paddle._C_ops.relu(batch_norm__30)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16, 64x64x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu_4, parameter_32, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x64x-1x-1xf16, 256x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu_5, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        add_2 = paddle._C_ops.add(batch_norm__42, relu_3)

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_6 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x256x-1x-1xf16, 64x256x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu_6, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_7 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16, 64x64x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu_7, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        relu_8 = paddle._C_ops.relu(batch_norm__54)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x64x-1x-1xf16, 256x64x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu_8, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        add_3 = paddle._C_ops.add(batch_norm__60, relu_6)

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_9 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x256x-1x-1xf16, 128x256x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu_9, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_10 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16, 128x128x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu_10, parameter_62, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_63, parameter_64, parameter_65, parameter_66, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_11 = paddle._C_ops.relu(batch_norm__72)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x128x-1x-1xf16, 512x128x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu_11, parameter_67, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x256x-1x-1xf16, 512x256x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu_9, parameter_72, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        add_4 = paddle._C_ops.add(batch_norm__78, batch_norm__84)

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_12 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x512x-1x-1xf16, 128x512x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu_12, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_13 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16, 128x128x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu_13, parameter_82, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_83, parameter_84, parameter_85, parameter_86, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_14 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x128x-1x-1xf16, 512x128x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu_14, parameter_87, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        add_5 = paddle._C_ops.add(batch_norm__102, relu_12)

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_15 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x512x-1x-1xf16, 128x512x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu_15, parameter_92, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_93, parameter_94, parameter_95, parameter_96, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_16 = paddle._C_ops.relu(batch_norm__108)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16, 128x128x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu_16, parameter_97, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_98, parameter_99, parameter_100, parameter_101, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_17 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x128x-1x-1xf16, 512x128x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu_17, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_103, parameter_104, parameter_105, parameter_106, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        add_6 = paddle._C_ops.add(batch_norm__120, relu_15)

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_18 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x512x-1x-1xf16, 128x512x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu_18, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_108, parameter_109, parameter_110, parameter_111, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_19 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16, 128x128x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu_19, parameter_112, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_113, parameter_114, parameter_115, parameter_116, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_20 = paddle._C_ops.relu(batch_norm__132)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x128x-1x-1xf16, 512x128x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu_20, parameter_117, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        add_7 = paddle._C_ops.add(batch_norm__138, relu_18)

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_21 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x512x-1x-1xf16, 256x512x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu_21, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_123, parameter_124, parameter_125, parameter_126, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_22 = paddle._C_ops.relu(batch_norm__144)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu_22, parameter_127, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_128, parameter_129, parameter_130, parameter_131, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_23 = paddle._C_ops.relu(batch_norm__150)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu_23, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x512x-1x-1xf16, 1024x512x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu_21, parameter_137, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_138, parameter_139, parameter_140, parameter_141, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_8 = paddle._C_ops.add(batch_norm__156, batch_norm__162)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_24 = paddle._C_ops.relu(add_8)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu_24, parameter_142, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_143, parameter_144, parameter_145, parameter_146, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_25 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu_25, parameter_147, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_148, parameter_149, parameter_150, parameter_151, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_26 = paddle._C_ops.relu(batch_norm__174)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu_26, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_9 = paddle._C_ops.add(batch_norm__180, relu_24)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_27 = paddle._C_ops.relu(add_9)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu_27, parameter_157, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_28 = paddle._C_ops.relu(batch_norm__186)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu_28, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_29 = paddle._C_ops.relu(batch_norm__192)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu_29, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_10 = paddle._C_ops.add(batch_norm__198, relu_27)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_30 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu_30, parameter_172, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_173, parameter_174, parameter_175, parameter_176, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_31 = paddle._C_ops.relu(batch_norm__204)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu_31, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_32 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu_32, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_183, parameter_184, parameter_185, parameter_186, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_11 = paddle._C_ops.add(batch_norm__216, relu_30)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_33 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu_33, parameter_187, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_188, parameter_189, parameter_190, parameter_191, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_34 = paddle._C_ops.relu(batch_norm__222)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu_34, parameter_192, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_193, parameter_194, parameter_195, parameter_196, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_35 = paddle._C_ops.relu(batch_norm__228)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu_35, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_198, parameter_199, parameter_200, parameter_201, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_12 = paddle._C_ops.add(batch_norm__234, relu_33)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_36 = paddle._C_ops.relu(add_12)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu_36, parameter_202, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_203, parameter_204, parameter_205, parameter_206, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_37 = paddle._C_ops.relu(batch_norm__240)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu_37, parameter_207, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_208, parameter_209, parameter_210, parameter_211, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_38 = paddle._C_ops.relu(batch_norm__246)

        # pd_op.conv2d: (-1x1024x-1x-1xf16) <- (-1x256x-1x-1xf16, 1024x256x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu_38, parameter_212, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x-1x-1xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_213, parameter_214, parameter_215, parameter_216, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16, -1x1024x-1x-1xf16)
        add_13 = paddle._C_ops.add(batch_norm__252, relu_36)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_39 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x1024x-1x-1xf16, 512x1024x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu_39, parameter_217, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_218, parameter_219, parameter_220, parameter_221, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_40 = paddle._C_ops.relu(batch_norm__258)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu_40, parameter_222, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_223, parameter_224, parameter_225, parameter_226, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_41 = paddle._C_ops.relu(batch_norm__264)

        # pd_op.conv2d: (-1x2048x-1x-1xf16) <- (-1x512x-1x-1xf16, 2048x512x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu_41, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_228, parameter_229, parameter_230, parameter_231, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x2048x-1x-1xf16) <- (-1x1024x-1x-1xf16, 2048x1024x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu_39, parameter_232, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_233, parameter_234, parameter_235, parameter_236, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16, -1x2048x-1x-1xf16)
        add_14 = paddle._C_ops.add(batch_norm__270, batch_norm__276)

        # pd_op.relu: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16)
        relu_42 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x2048x-1x-1xf16, 512x2048x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu_42, parameter_237, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_238, parameter_239, parameter_240, parameter_241, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_43 = paddle._C_ops.relu(batch_norm__282)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu_43, parameter_242, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_243, parameter_244, parameter_245, parameter_246, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_44 = paddle._C_ops.relu(batch_norm__288)

        # pd_op.conv2d: (-1x2048x-1x-1xf16) <- (-1x512x-1x-1xf16, 2048x512x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu_44, parameter_247, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_248, parameter_249, parameter_250, parameter_251, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16, -1x2048x-1x-1xf16)
        add_15 = paddle._C_ops.add(batch_norm__294, relu_42)

        # pd_op.relu: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16)
        relu_45 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x2048x-1x-1xf16, 512x2048x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(relu_45, parameter_252, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_253, parameter_254, parameter_255, parameter_256, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_46 = paddle._C_ops.relu(batch_norm__300)

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x3x3xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu_46, parameter_257, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_258, parameter_259, parameter_260, parameter_261, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_47 = paddle._C_ops.relu(batch_norm__306)

        # pd_op.conv2d: (-1x2048x-1x-1xf16) <- (-1x512x-1x-1xf16, 2048x512x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu_47, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x-1x-1xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_263, parameter_264, parameter_265, parameter_266, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16, -1x2048x-1x-1xf16)
        add_16 = paddle._C_ops.add(batch_norm__312, relu_45)

        # pd_op.relu: (-1x2048x-1x-1xf16) <- (-1x2048x-1x-1xf16)
        relu_48 = paddle._C_ops.relu(add_16)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x512x-1x-1xf16, 256x512x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu_21, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_268, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_17 = paddle._C_ops.add(conv2d_53, reshape_0)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x1024x-1x-1xf16, 256x1024x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu_39, parameter_269, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_270, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_18 = paddle._C_ops.add(conv2d_54, reshape_2)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x2048x-1x-1xf16, 256x2048x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(relu_48, parameter_271, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_272, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_19 = paddle._C_ops.add(conv2d_55, reshape_4)

        # pd_op.nearest_interp: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(add_19, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        add_20 = paddle._C_ops.add(add_18, nearest_interp_0)

        # pd_op.nearest_interp: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(add_20, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        add_21 = paddle._C_ops.add(add_17, nearest_interp_1)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(add_21, parameter_273, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_274, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_22 = paddle._C_ops.add(conv2d_56, reshape_6)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(add_20, parameter_275, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_276, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_23 = paddle._C_ops.add(conv2d_57, reshape_8)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(add_19, parameter_277, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_278, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_24 = paddle._C_ops.add(conv2d_58, reshape_10)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(add_24, parameter_279, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_280, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_25 = paddle._C_ops.add(conv2d_59, reshape_12)

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_49 = paddle._C_ops.relu(add_25)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_60 = paddle._C_ops.conv2d(relu_49, parameter_281, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_282, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_26 = paddle._C_ops.add(conv2d_60, reshape_14)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_61 = paddle._C_ops.conv2d(add_22, parameter_283, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_284, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_27 = paddle._C_ops.add(conv2d_61, reshape_16)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_0, group_norm_1, group_norm_2 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_27, parameter_285, parameter_286, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_50 = paddle._C_ops.relu(group_norm_0)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(add_22, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_28 = paddle._C_ops.add(conv2d_62, reshape_18)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_3, group_norm_4, group_norm_5 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_28, parameter_289, parameter_290, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_51 = paddle._C_ops.relu(group_norm_3)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_63 = paddle._C_ops.conv2d(relu_50, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_292, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_29 = paddle._C_ops.add(conv2d_63, reshape_20)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_6, group_norm_7, group_norm_8 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_29, parameter_293, parameter_294, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_52 = paddle._C_ops.relu(group_norm_6)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu_51, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_296, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_30 = paddle._C_ops.add(conv2d_64, reshape_22)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_9, group_norm_10, group_norm_11 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_30, parameter_297, parameter_298, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_53 = paddle._C_ops.relu(group_norm_9)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_65 = paddle._C_ops.conv2d(relu_52, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_31 = paddle._C_ops.add(conv2d_65, reshape_24)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_12, group_norm_13, group_norm_14 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_31, parameter_301, parameter_302, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_54 = paddle._C_ops.relu(group_norm_12)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu_53, parameter_303, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_32 = paddle._C_ops.add(conv2d_66, reshape_26)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_15, group_norm_16, group_norm_17 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_32, parameter_305, parameter_306, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_55 = paddle._C_ops.relu(group_norm_15)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu_54, parameter_307, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_308, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_33 = paddle._C_ops.add(conv2d_67, reshape_28)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_18, group_norm_19, group_norm_20 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_33, parameter_309, parameter_310, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_56 = paddle._C_ops.relu(group_norm_18)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(relu_55, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_312, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_34 = paddle._C_ops.add(conv2d_68, reshape_30)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_21, group_norm_22, group_norm_23 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_34, parameter_313, parameter_314, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_57 = paddle._C_ops.relu(group_norm_21)

        # pd_op.conv2d: (-1x80x-1x-1xf16) <- (-1x256x-1x-1xf16, 80x256x3x3xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu_56, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf16, 0x80xf16) <- (80xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16, 1x80x1x1xf16)
        add_35 = paddle._C_ops.add(conv2d_69, reshape_32)

        # pd_op.conv2d: (-1x4x-1x-1xf16) <- (-1x256x-1x-1xf16, 4x256x3x3xf16)
        conv2d_70 = paddle._C_ops.conv2d(relu_57, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1x4x1x1xf16)
        add_36 = paddle._C_ops.add(conv2d_70, reshape_34)

        # pd_op.multiply: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf16)
        multiply_1 = paddle._C_ops.multiply(add_36, parameter_319)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x3x3xf16)
        conv2d_71 = paddle._C_ops.conv2d(relu_57, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_37 = paddle._C_ops.add(conv2d_71, reshape_36)

        # pd_op.relu: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16)
        relu_58 = paddle._C_ops.relu(multiply_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf32)
        scale_0 = paddle._C_ops.scale(relu_58, full_0, float('0'), True)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(add_23, parameter_283, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_284, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_38 = paddle._C_ops.add(conv2d_72, reshape_38)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_24, group_norm_25, group_norm_26 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_38, parameter_285, parameter_286, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_59 = paddle._C_ops.relu(group_norm_24)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_73 = paddle._C_ops.conv2d(add_23, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_39 = paddle._C_ops.add(conv2d_73, reshape_40)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_27, group_norm_28, group_norm_29 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_39, parameter_289, parameter_290, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_60 = paddle._C_ops.relu(group_norm_27)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_74 = paddle._C_ops.conv2d(relu_59, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_292, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_40 = paddle._C_ops.add(conv2d_74, reshape_42)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_30, group_norm_31, group_norm_32 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_40, parameter_293, parameter_294, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_61 = paddle._C_ops.relu(group_norm_30)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_75 = paddle._C_ops.conv2d(relu_60, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_296, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_41 = paddle._C_ops.add(conv2d_75, reshape_44)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_33, group_norm_34, group_norm_35 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_41, parameter_297, parameter_298, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_62 = paddle._C_ops.relu(group_norm_33)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu_61, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_42 = paddle._C_ops.add(conv2d_76, reshape_46)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_36, group_norm_37, group_norm_38 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_42, parameter_301, parameter_302, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_63 = paddle._C_ops.relu(group_norm_36)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_77 = paddle._C_ops.conv2d(relu_62, parameter_303, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_43 = paddle._C_ops.add(conv2d_77, reshape_48)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_39, group_norm_40, group_norm_41 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_43, parameter_305, parameter_306, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_64 = paddle._C_ops.relu(group_norm_39)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_78 = paddle._C_ops.conv2d(relu_63, parameter_307, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_308, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_44 = paddle._C_ops.add(conv2d_78, reshape_50)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_42, group_norm_43, group_norm_44 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_44, parameter_309, parameter_310, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_65 = paddle._C_ops.relu(group_norm_42)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_79 = paddle._C_ops.conv2d(relu_64, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_312, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_45 = paddle._C_ops.add(conv2d_79, reshape_52)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_45, group_norm_46, group_norm_47 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_45, parameter_313, parameter_314, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_66 = paddle._C_ops.relu(group_norm_45)

        # pd_op.conv2d: (-1x80x-1x-1xf16) <- (-1x256x-1x-1xf16, 80x256x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(relu_65, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf16, 0x80xf16) <- (80xf16, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16, 1x80x1x1xf16)
        add_46 = paddle._C_ops.add(conv2d_80, reshape_54)

        # pd_op.conv2d: (-1x4x-1x-1xf16) <- (-1x256x-1x-1xf16, 4x256x3x3xf16)
        conv2d_81 = paddle._C_ops.conv2d(relu_66, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1x4x1x1xf16)
        add_47 = paddle._C_ops.add(conv2d_81, reshape_56)

        # pd_op.multiply: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf16)
        multiply_2 = paddle._C_ops.multiply(add_47, parameter_322)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x3x3xf16)
        conv2d_82 = paddle._C_ops.conv2d(relu_66, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_48 = paddle._C_ops.add(conv2d_82, reshape_58)

        # pd_op.relu: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16)
        relu_67 = paddle._C_ops.relu(multiply_2)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf32)
        scale_1 = paddle._C_ops.scale(relu_67, full_1, float('0'), True)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_83 = paddle._C_ops.conv2d(add_24, parameter_283, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_284, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_49 = paddle._C_ops.add(conv2d_83, reshape_60)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_48, group_norm_49, group_norm_50 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_49, parameter_285, parameter_286, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_68 = paddle._C_ops.relu(group_norm_48)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_84 = paddle._C_ops.conv2d(add_24, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_50 = paddle._C_ops.add(conv2d_84, reshape_62)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_51, group_norm_52, group_norm_53 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_50, parameter_289, parameter_290, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_69 = paddle._C_ops.relu(group_norm_51)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu_68, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_292, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_51 = paddle._C_ops.add(conv2d_85, reshape_64)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_54, group_norm_55, group_norm_56 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_51, parameter_293, parameter_294, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_70 = paddle._C_ops.relu(group_norm_54)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_86 = paddle._C_ops.conv2d(relu_69, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_296, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_52 = paddle._C_ops.add(conv2d_86, reshape_66)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_57, group_norm_58, group_norm_59 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_52, parameter_297, parameter_298, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_71 = paddle._C_ops.relu(group_norm_57)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_87 = paddle._C_ops.conv2d(relu_70, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_53 = paddle._C_ops.add(conv2d_87, reshape_68)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_60, group_norm_61, group_norm_62 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_53, parameter_301, parameter_302, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_72 = paddle._C_ops.relu(group_norm_60)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_88 = paddle._C_ops.conv2d(relu_71, parameter_303, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_54 = paddle._C_ops.add(conv2d_88, reshape_70)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_63, group_norm_64, group_norm_65 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_54, parameter_305, parameter_306, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_73 = paddle._C_ops.relu(group_norm_63)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_89 = paddle._C_ops.conv2d(relu_72, parameter_307, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_308, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_55 = paddle._C_ops.add(conv2d_89, reshape_72)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_66, group_norm_67, group_norm_68 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_55, parameter_309, parameter_310, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_74 = paddle._C_ops.relu(group_norm_66)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_90 = paddle._C_ops.conv2d(relu_73, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_312, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_56 = paddle._C_ops.add(conv2d_90, reshape_74)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_69, group_norm_70, group_norm_71 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_56, parameter_313, parameter_314, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_75 = paddle._C_ops.relu(group_norm_69)

        # pd_op.conv2d: (-1x80x-1x-1xf16) <- (-1x256x-1x-1xf16, 80x256x3x3xf16)
        conv2d_91 = paddle._C_ops.conv2d(relu_74, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf16, 0x80xf16) <- (80xf16, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16, 1x80x1x1xf16)
        add_57 = paddle._C_ops.add(conv2d_91, reshape_76)

        # pd_op.conv2d: (-1x4x-1x-1xf16) <- (-1x256x-1x-1xf16, 4x256x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(relu_75, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1x4x1x1xf16)
        add_58 = paddle._C_ops.add(conv2d_92, reshape_78)

        # pd_op.multiply: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf16)
        multiply_3 = paddle._C_ops.multiply(add_58, parameter_323)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x3x3xf16)
        conv2d_93 = paddle._C_ops.conv2d(relu_75, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_59 = paddle._C_ops.add(conv2d_93, reshape_80)

        # pd_op.relu: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16)
        relu_76 = paddle._C_ops.relu(multiply_3)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf32)
        scale_2 = paddle._C_ops.scale(relu_76, full_2, float('0'), True)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_94 = paddle._C_ops.conv2d(add_25, parameter_283, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_284, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_60 = paddle._C_ops.add(conv2d_94, reshape_82)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_72, group_norm_73, group_norm_74 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_60, parameter_285, parameter_286, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_77 = paddle._C_ops.relu(group_norm_72)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_95 = paddle._C_ops.conv2d(add_25, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_61 = paddle._C_ops.add(conv2d_95, reshape_84)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_75, group_norm_76, group_norm_77 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_61, parameter_289, parameter_290, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_78 = paddle._C_ops.relu(group_norm_75)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_96 = paddle._C_ops.conv2d(relu_77, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_292, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_62 = paddle._C_ops.add(conv2d_96, reshape_86)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_78, group_norm_79, group_norm_80 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_62, parameter_293, parameter_294, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_79 = paddle._C_ops.relu(group_norm_78)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_97 = paddle._C_ops.conv2d(relu_78, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_296, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_63 = paddle._C_ops.add(conv2d_97, reshape_88)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_81, group_norm_82, group_norm_83 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_63, parameter_297, parameter_298, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_80 = paddle._C_ops.relu(group_norm_81)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_98 = paddle._C_ops.conv2d(relu_79, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_64 = paddle._C_ops.add(conv2d_98, reshape_90)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_84, group_norm_85, group_norm_86 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_64, parameter_301, parameter_302, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_81 = paddle._C_ops.relu(group_norm_84)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_99 = paddle._C_ops.conv2d(relu_80, parameter_303, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_65 = paddle._C_ops.add(conv2d_99, reshape_92)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_87, group_norm_88, group_norm_89 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_65, parameter_305, parameter_306, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_82 = paddle._C_ops.relu(group_norm_87)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_100 = paddle._C_ops.conv2d(relu_81, parameter_307, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_308, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_66 = paddle._C_ops.add(conv2d_100, reshape_94)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_90, group_norm_91, group_norm_92 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_66, parameter_309, parameter_310, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_83 = paddle._C_ops.relu(group_norm_90)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_101 = paddle._C_ops.conv2d(relu_82, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_312, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_67 = paddle._C_ops.add(conv2d_101, reshape_96)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_93, group_norm_94, group_norm_95 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_67, parameter_313, parameter_314, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_84 = paddle._C_ops.relu(group_norm_93)

        # pd_op.conv2d: (-1x80x-1x-1xf16) <- (-1x256x-1x-1xf16, 80x256x3x3xf16)
        conv2d_102 = paddle._C_ops.conv2d(relu_83, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf16, 0x80xf16) <- (80xf16, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16, 1x80x1x1xf16)
        add_68 = paddle._C_ops.add(conv2d_102, reshape_98)

        # pd_op.conv2d: (-1x4x-1x-1xf16) <- (-1x256x-1x-1xf16, 4x256x3x3xf16)
        conv2d_103 = paddle._C_ops.conv2d(relu_84, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1x4x1x1xf16)
        add_69 = paddle._C_ops.add(conv2d_103, reshape_100)

        # pd_op.multiply: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf16)
        multiply_4 = paddle._C_ops.multiply(add_69, parameter_324)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x3x3xf16)
        conv2d_104 = paddle._C_ops.conv2d(relu_84, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_70 = paddle._C_ops.add(conv2d_104, reshape_102)

        # pd_op.relu: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16)
        relu_85 = paddle._C_ops.relu(multiply_4)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('64'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf32)
        scale_3 = paddle._C_ops.scale(relu_85, full_3, float('0'), True)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_105 = paddle._C_ops.conv2d(add_26, parameter_283, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_284, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_71 = paddle._C_ops.add(conv2d_105, reshape_104)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_96, group_norm_97, group_norm_98 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_71, parameter_285, parameter_286, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_86 = paddle._C_ops.relu(group_norm_96)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_106 = paddle._C_ops.conv2d(add_26, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_72 = paddle._C_ops.add(conv2d_106, reshape_106)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_99, group_norm_100, group_norm_101 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_72, parameter_289, parameter_290, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_87 = paddle._C_ops.relu(group_norm_99)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_107 = paddle._C_ops.conv2d(relu_86, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_292, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_73 = paddle._C_ops.add(conv2d_107, reshape_108)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_102, group_norm_103, group_norm_104 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_73, parameter_293, parameter_294, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_88 = paddle._C_ops.relu(group_norm_102)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_108 = paddle._C_ops.conv2d(relu_87, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_296, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_74 = paddle._C_ops.add(conv2d_108, reshape_110)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_105, group_norm_106, group_norm_107 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_74, parameter_297, parameter_298, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_89 = paddle._C_ops.relu(group_norm_105)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_109 = paddle._C_ops.conv2d(relu_88, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_75 = paddle._C_ops.add(conv2d_109, reshape_112)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_108, group_norm_109, group_norm_110 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_75, parameter_301, parameter_302, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_90 = paddle._C_ops.relu(group_norm_108)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_110 = paddle._C_ops.conv2d(relu_89, parameter_303, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_76 = paddle._C_ops.add(conv2d_110, reshape_114)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_111, group_norm_112, group_norm_113 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_76, parameter_305, parameter_306, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_91 = paddle._C_ops.relu(group_norm_111)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_111 = paddle._C_ops.conv2d(relu_90, parameter_307, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_308, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_77 = paddle._C_ops.add(conv2d_111, reshape_116)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_114, group_norm_115, group_norm_116 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_77, parameter_309, parameter_310, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_92 = paddle._C_ops.relu(group_norm_114)

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 256x256x3x3xf16)
        conv2d_112 = paddle._C_ops.conv2d(relu_91, parameter_311, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_312, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x1x1xf16)
        add_78 = paddle._C_ops.add(conv2d_112, reshape_118)

        # pd_op.group_norm: (-1x256x-1x-1xf16, -1x32xf32, -1x32xf32) <- (-1x256x-1x-1xf16, 256xf16, 256xf16)
        group_norm_117, group_norm_118, group_norm_119 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add_78, parameter_313, parameter_314, float('1e-05'), 32, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_93 = paddle._C_ops.relu(group_norm_117)

        # pd_op.conv2d: (-1x80x-1x-1xf16) <- (-1x256x-1x-1xf16, 80x256x3x3xf16)
        conv2d_113 = paddle._C_ops.conv2d(relu_92, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf16, 0x80xf16) <- (80xf16, 4xi64)
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16, 1x80x1x1xf16)
        add_79 = paddle._C_ops.add(conv2d_113, reshape_120)

        # pd_op.conv2d: (-1x4x-1x-1xf16) <- (-1x256x-1x-1xf16, 4x256x3x3xf16)
        conv2d_114 = paddle._C_ops.conv2d(relu_93, parameter_317, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1x4x1x1xf16)
        add_80 = paddle._C_ops.add(conv2d_114, reshape_122)

        # pd_op.multiply: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf16)
        multiply_5 = paddle._C_ops.multiply(add_80, parameter_325)

        # pd_op.conv2d: (-1x1x-1x-1xf16) <- (-1x256x-1x-1xf16, 1x256x3x3xf16)
        conv2d_115 = paddle._C_ops.conv2d(relu_93, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf16, 0x1xf16) <- (1xf16, 4xi64)
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16, 1x1x1x1xf16)
        add_81 = paddle._C_ops.add(conv2d_115, reshape_124)

        # pd_op.relu: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16)
        relu_94 = paddle._C_ops.relu(multiply_5)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('128'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x4x-1x-1xf16) <- (-1x4x-1x-1xf16, 1xf32)
        scale_4 = paddle._C_ops.scale(relu_94, full_4, float('0'), True)

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(add_22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [3]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(add_22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [4]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_1, full_5, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_6 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_1 = paddle._C_ops.cast(scale_5, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_7 = paddle._C_ops.full([1], float('8'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_0 = paddle.arange(full_6, cast_1, full_7, dtype='int64')

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_0, full_8, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_9 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_2 = paddle._C_ops.cast(scale_6, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_10 = paddle._C_ops.full([1], float('8'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_1 = paddle.arange(full_9, cast_2, full_10, dtype='int64')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.unsqueeze_: (1x-1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_0, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_1, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_0 = [slice_0, slice_1]

        # pd_op.expand: (-1x-1xi64) <- (1x-1xi64, [1xi32, 1xi32])
        expand_0 = paddle._C_ops.expand(unsqueeze__0, [x.reshape([]) for x in combine_0])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_1 = [slice_0, slice_1]

        # pd_op.expand: (-1x-1xi64) <- (-1x1xi64, [1xi32, 1xi32])
        expand_1 = paddle._C_ops.expand(unsqueeze__2, [x.reshape([]) for x in combine_1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_0, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_1, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1xi64, -1xi64]) <- (-1xi64, -1xi64)
        combine_2 = [reshape__0, reshape__2]

        # pd_op.stack: (-1x2xi64) <- ([-1xi64, -1xi64])
        stack_0 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.cast: (-1x2xf16) <- (-1x2xi64)
        cast_3 = paddle._C_ops.cast(stack_0, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2xf16) <- (-1x2xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(cast_3, full_11, float('4'), True)

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add_23, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [3]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(add_23, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [4]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_74, full_int_array_75, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_3, full_12, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_13 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_4 = paddle._C_ops.cast(scale_7, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_14 = paddle._C_ops.full([1], float('16'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_2 = paddle.arange(full_13, cast_4, full_14, dtype='int64')

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_2, full_15, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_16 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_5 = paddle._C_ops.cast(scale_8, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_17 = paddle._C_ops.full([1], float('16'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_3 = paddle.arange(full_16, cast_5, full_17, dtype='int64')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [0]

        # pd_op.unsqueeze_: (1x-1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_2, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_3, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_3 = [slice_2, slice_3]

        # pd_op.expand: (-1x-1xi64) <- (1x-1xi64, [1xi32, 1xi32])
        expand_2 = paddle._C_ops.expand(unsqueeze__4, [x.reshape([]) for x in combine_3])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_4 = [slice_2, slice_3]

        # pd_op.expand: (-1x-1xi64) <- (-1x1xi64, [1xi32, 1xi32])
        expand_3 = paddle._C_ops.expand(unsqueeze__6, [x.reshape([]) for x in combine_4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_2, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_3, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1xi64, -1xi64]) <- (-1xi64, -1xi64)
        combine_5 = [reshape__4, reshape__6]

        # pd_op.stack: (-1x2xi64) <- ([-1xi64, -1xi64])
        stack_1 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.cast: (-1x2xf16) <- (-1x2xi64)
        cast_6 = paddle._C_ops.cast(stack_1, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2xf16) <- (-1x2xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(cast_6, full_18, float('8'), True)

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(add_24, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [3]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add_24, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [4]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_5, full_19, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_20 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_7 = paddle._C_ops.cast(scale_9, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_21 = paddle._C_ops.full([1], float('32'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_4 = paddle.arange(full_20, cast_7, full_21, dtype='int64')

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_4, full_22, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_23 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_8 = paddle._C_ops.cast(scale_10, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_24 = paddle._C_ops.full([1], float('32'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_5 = paddle.arange(full_23, cast_8, full_24, dtype='int64')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.unsqueeze_: (1x-1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_4, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_5, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_6 = [slice_4, slice_5]

        # pd_op.expand: (-1x-1xi64) <- (1x-1xi64, [1xi32, 1xi32])
        expand_4 = paddle._C_ops.expand(unsqueeze__8, [x.reshape([]) for x in combine_6])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_7 = [slice_4, slice_5]

        # pd_op.expand: (-1x-1xi64) <- (-1x1xi64, [1xi32, 1xi32])
        expand_5 = paddle._C_ops.expand(unsqueeze__10, [x.reshape([]) for x in combine_7])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_4, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_5, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1xi64, -1xi64]) <- (-1xi64, -1xi64)
        combine_8 = [reshape__8, reshape__10]

        # pd_op.stack: (-1x2xi64) <- ([-1xi64, -1xi64])
        stack_2 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.cast: (-1x2xf16) <- (-1x2xi64)
        cast_9 = paddle._C_ops.cast(stack_2, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2xf16) <- (-1x2xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(cast_9, full_25, float('16'), True)

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(add_25, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [3]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(add_25, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [4]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full([1], float('64'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_7, full_26, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_27 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_10 = paddle._C_ops.cast(scale_11, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_28 = paddle._C_ops.full([1], float('64'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_6 = paddle.arange(full_27, cast_10, full_28, dtype='int64')

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('64'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_6, full_29, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_30 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_11 = paddle._C_ops.cast(scale_12, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_31 = paddle._C_ops.full([1], float('64'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_7 = paddle.arange(full_30, cast_11, full_31, dtype='int64')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.unsqueeze_: (1x-1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_6, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_7, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_9 = [slice_6, slice_7]

        # pd_op.expand: (-1x-1xi64) <- (1x-1xi64, [1xi32, 1xi32])
        expand_6 = paddle._C_ops.expand(unsqueeze__12, [x.reshape([]) for x in combine_9])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_10 = [slice_6, slice_7]

        # pd_op.expand: (-1x-1xi64) <- (-1x1xi64, [1xi32, 1xi32])
        expand_7 = paddle._C_ops.expand(unsqueeze__14, [x.reshape([]) for x in combine_10])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_6, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_7, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1xi64, -1xi64]) <- (-1xi64, -1xi64)
        combine_11 = [reshape__12, reshape__14]

        # pd_op.stack: (-1x2xi64) <- ([-1xi64, -1xi64])
        stack_3 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.cast: (-1x2xf16) <- (-1x2xi64)
        cast_12 = paddle._C_ops.cast(stack_3, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2xf16) <- (-1x2xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(cast_12, full_32, float('32'), True)

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add_26, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [3]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.shape: (4xi32) <- (-1x256x-1x-1xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(add_26, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [4]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_9, [0], full_int_array_98, full_int_array_99, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_33 = paddle._C_ops.full([1], float('128'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_13 = paddle._C_ops.scale(slice_9, full_33, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_34 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_13 = paddle._C_ops.cast(scale_13, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_35 = paddle._C_ops.full([1], float('128'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_8 = paddle.arange(full_34, cast_13, full_35, dtype='int64')

        # pd_op.full: (1xf32) <- ()
        full_36 = paddle._C_ops.full([1], float('128'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1xi32) <- (1xi32, 1xf32)
        scale_14 = paddle._C_ops.scale(slice_8, full_36, float('0'), True)

        # pd_op.full: (1xi64) <- ()
        full_37 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_14 = paddle._C_ops.cast(scale_14, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_38 = paddle._C_ops.full([1], float('128'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, 1xi64, 1xi64)
        arange_9 = paddle.arange(full_37, cast_14, full_38, dtype='int64')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [0]

        # pd_op.unsqueeze_: (1x-1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_8, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [1]

        # pd_op.unsqueeze_: (-1x1xi64, None) <- (-1xi64, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(arange_9, full_int_array_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_12 = [slice_8, slice_9]

        # pd_op.expand: (-1x-1xi64) <- (1x-1xi64, [1xi32, 1xi32])
        expand_8 = paddle._C_ops.expand(unsqueeze__16, [x.reshape([]) for x in combine_12])

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_13 = [slice_8, slice_9]

        # pd_op.expand: (-1x-1xi64) <- (-1x1xi64, [1xi32, 1xi32])
        expand_9 = paddle._C_ops.expand(unsqueeze__18, [x.reshape([]) for x in combine_13])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_8, full_int_array_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [-1]

        # pd_op.reshape_: (-1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(expand_9, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1xi64, -1xi64]) <- (-1xi64, -1xi64)
        combine_14 = [reshape__16, reshape__18]

        # pd_op.stack: (-1x2xi64) <- ([-1xi64, -1xi64])
        stack_4 = paddle._C_ops.stack(combine_14, -1)

        # pd_op.cast: (-1x2xf16) <- (-1x2xi64)
        cast_15 = paddle._C_ops.cast(stack_4, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2xf16) <- (-1x2xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(cast_15, full_39, float('64'), True)

        # pd_op.sigmoid: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16)
        sigmoid_0 = paddle._C_ops.sigmoid(add_35)

        # pd_op.flatten: (-1x80x-1xf16, None) <- (-1x80x-1x-1xf16)
        flatten_0, flatten_1 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x80xf16) <- (-1x80x-1xf16)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_1 = paddle._C_ops.sigmoid(add_37)

        # pd_op.flatten: (-1x1x-1xf16, None) <- (-1x1x-1x-1xf16)
        flatten_2, flatten_3 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_1, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x1xf16) <- (-1x1x-1xf16)
        transpose_1 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])

        # pd_op.multiply: (-1x-1x80xf16) <- (-1x-1x80xf16, -1x-1x1xf16)
        multiply_6 = paddle._C_ops.multiply(transpose_0, transpose_1)

        # pd_op.flatten: (-1x4x-1xf16, None) <- (-1x4x-1x-1xf16)
        flatten_4, flatten_5 = (lambda x, f: f(x))(paddle._C_ops.flatten(scale_0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_2 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(scale__0, [1], full_int_array_104, full_int_array_105, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [1]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_2, [2], full_int_array_106, full_int_array_107, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_0 = paddle._C_ops.subtract(slice_10, slice_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(scale__0, [1], full_int_array_108, full_int_array_109, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [2]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_2, [2], full_int_array_110, full_int_array_111, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_1 = paddle._C_ops.subtract(slice_12, slice_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(scale__0, [1], full_int_array_112, full_int_array_113, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [3]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_2, [2], full_int_array_114, full_int_array_115, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_82 = paddle._C_ops.add(slice_14, slice_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(scale__0, [1], full_int_array_116, full_int_array_117, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [4]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_2, [2], full_int_array_118, full_int_array_119, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_83 = paddle._C_ops.add(slice_16, slice_17)

        # builtin.combine: ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16]) <- (-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16)
        combine_15 = [subtract_0, subtract_1, add_82, add_83]

        # pd_op.stack: (-1x4x-1xf16) <- ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16])
        stack_5 = paddle._C_ops.stack(combine_15, 1)

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_3 = paddle._C_ops.transpose(stack_5, [0, 2, 1])

        # pd_op.sigmoid: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16)
        sigmoid_2 = paddle._C_ops.sigmoid(add_46)

        # pd_op.flatten: (-1x80x-1xf16, None) <- (-1x80x-1x-1xf16)
        flatten_6, flatten_7 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_2, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x80xf16) <- (-1x80x-1xf16)
        transpose_4 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_3 = paddle._C_ops.sigmoid(add_48)

        # pd_op.flatten: (-1x1x-1xf16, None) <- (-1x1x-1x-1xf16)
        flatten_8, flatten_9 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_3, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x1xf16) <- (-1x1x-1xf16)
        transpose_5 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])

        # pd_op.multiply: (-1x-1x80xf16) <- (-1x-1x80xf16, -1x-1x1xf16)
        multiply_7 = paddle._C_ops.multiply(transpose_4, transpose_5)

        # pd_op.flatten: (-1x4x-1xf16, None) <- (-1x4x-1x-1xf16)
        flatten_10, flatten_11 = (lambda x, f: f(x))(paddle._C_ops.flatten(scale_1, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_6 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(scale__1, [1], full_int_array_120, full_int_array_121, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [1]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_6, [2], full_int_array_122, full_int_array_123, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_2 = paddle._C_ops.subtract(slice_18, slice_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(scale__1, [1], full_int_array_124, full_int_array_125, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [2]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_6, [2], full_int_array_126, full_int_array_127, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_3 = paddle._C_ops.subtract(slice_20, slice_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(scale__1, [1], full_int_array_128, full_int_array_129, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [3]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_6, [2], full_int_array_130, full_int_array_131, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_84 = paddle._C_ops.add(slice_22, slice_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(scale__1, [1], full_int_array_132, full_int_array_133, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [4]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_6, [2], full_int_array_134, full_int_array_135, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_85 = paddle._C_ops.add(slice_24, slice_25)

        # builtin.combine: ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16]) <- (-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16)
        combine_16 = [subtract_2, subtract_3, add_84, add_85]

        # pd_op.stack: (-1x4x-1xf16) <- ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16])
        stack_6 = paddle._C_ops.stack(combine_16, 1)

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_7 = paddle._C_ops.transpose(stack_6, [0, 2, 1])

        # pd_op.sigmoid: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16)
        sigmoid_4 = paddle._C_ops.sigmoid(add_57)

        # pd_op.flatten: (-1x80x-1xf16, None) <- (-1x80x-1x-1xf16)
        flatten_12, flatten_13 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_4, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x80xf16) <- (-1x80x-1xf16)
        transpose_8 = paddle._C_ops.transpose(flatten_12, [0, 2, 1])

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_5 = paddle._C_ops.sigmoid(add_59)

        # pd_op.flatten: (-1x1x-1xf16, None) <- (-1x1x-1x-1xf16)
        flatten_14, flatten_15 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_5, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x1xf16) <- (-1x1x-1xf16)
        transpose_9 = paddle._C_ops.transpose(flatten_14, [0, 2, 1])

        # pd_op.multiply: (-1x-1x80xf16) <- (-1x-1x80xf16, -1x-1x1xf16)
        multiply_8 = paddle._C_ops.multiply(transpose_8, transpose_9)

        # pd_op.flatten: (-1x4x-1xf16, None) <- (-1x4x-1x-1xf16)
        flatten_16, flatten_17 = (lambda x, f: f(x))(paddle._C_ops.flatten(scale_2, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_10 = paddle._C_ops.transpose(flatten_16, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(scale__2, [1], full_int_array_136, full_int_array_137, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [1]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_10, [2], full_int_array_138, full_int_array_139, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_4 = paddle._C_ops.subtract(slice_26, slice_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(scale__2, [1], full_int_array_140, full_int_array_141, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [2]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_10, [2], full_int_array_142, full_int_array_143, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_5 = paddle._C_ops.subtract(slice_28, slice_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(scale__2, [1], full_int_array_144, full_int_array_145, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [3]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_10, [2], full_int_array_146, full_int_array_147, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_86 = paddle._C_ops.add(slice_30, slice_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(scale__2, [1], full_int_array_148, full_int_array_149, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [4]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_10, [2], full_int_array_150, full_int_array_151, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_87 = paddle._C_ops.add(slice_32, slice_33)

        # builtin.combine: ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16]) <- (-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16)
        combine_17 = [subtract_4, subtract_5, add_86, add_87]

        # pd_op.stack: (-1x4x-1xf16) <- ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16])
        stack_7 = paddle._C_ops.stack(combine_17, 1)

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_11 = paddle._C_ops.transpose(stack_7, [0, 2, 1])

        # pd_op.sigmoid: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16)
        sigmoid_6 = paddle._C_ops.sigmoid(add_68)

        # pd_op.flatten: (-1x80x-1xf16, None) <- (-1x80x-1x-1xf16)
        flatten_18, flatten_19 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_6, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x80xf16) <- (-1x80x-1xf16)
        transpose_12 = paddle._C_ops.transpose(flatten_18, [0, 2, 1])

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_7 = paddle._C_ops.sigmoid(add_70)

        # pd_op.flatten: (-1x1x-1xf16, None) <- (-1x1x-1x-1xf16)
        flatten_20, flatten_21 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_7, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x1xf16) <- (-1x1x-1xf16)
        transpose_13 = paddle._C_ops.transpose(flatten_20, [0, 2, 1])

        # pd_op.multiply: (-1x-1x80xf16) <- (-1x-1x80xf16, -1x-1x1xf16)
        multiply_9 = paddle._C_ops.multiply(transpose_12, transpose_13)

        # pd_op.flatten: (-1x4x-1xf16, None) <- (-1x4x-1x-1xf16)
        flatten_22, flatten_23 = (lambda x, f: f(x))(paddle._C_ops.flatten(scale_3, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_14 = paddle._C_ops.transpose(flatten_22, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(scale__3, [1], full_int_array_152, full_int_array_153, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [1]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_14, [2], full_int_array_154, full_int_array_155, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_6 = paddle._C_ops.subtract(slice_34, slice_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(scale__3, [1], full_int_array_156, full_int_array_157, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [2]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_14, [2], full_int_array_158, full_int_array_159, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_7 = paddle._C_ops.subtract(slice_36, slice_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(scale__3, [1], full_int_array_160, full_int_array_161, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [3]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_14, [2], full_int_array_162, full_int_array_163, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_88 = paddle._C_ops.add(slice_38, slice_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(scale__3, [1], full_int_array_164, full_int_array_165, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [4]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_14, [2], full_int_array_166, full_int_array_167, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_89 = paddle._C_ops.add(slice_40, slice_41)

        # builtin.combine: ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16]) <- (-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16)
        combine_18 = [subtract_6, subtract_7, add_88, add_89]

        # pd_op.stack: (-1x4x-1xf16) <- ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16])
        stack_8 = paddle._C_ops.stack(combine_18, 1)

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_15 = paddle._C_ops.transpose(stack_8, [0, 2, 1])

        # pd_op.sigmoid: (-1x80x-1x-1xf16) <- (-1x80x-1x-1xf16)
        sigmoid_8 = paddle._C_ops.sigmoid(add_79)

        # pd_op.flatten: (-1x80x-1xf16, None) <- (-1x80x-1x-1xf16)
        flatten_24, flatten_25 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_8, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x80xf16) <- (-1x80x-1xf16)
        transpose_16 = paddle._C_ops.transpose(flatten_24, [0, 2, 1])

        # pd_op.sigmoid: (-1x1x-1x-1xf16) <- (-1x1x-1x-1xf16)
        sigmoid_9 = paddle._C_ops.sigmoid(add_81)

        # pd_op.flatten: (-1x1x-1xf16, None) <- (-1x1x-1x-1xf16)
        flatten_26, flatten_27 = (lambda x, f: f(x))(paddle._C_ops.flatten(sigmoid_9, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x1xf16) <- (-1x1x-1xf16)
        transpose_17 = paddle._C_ops.transpose(flatten_26, [0, 2, 1])

        # pd_op.multiply: (-1x-1x80xf16) <- (-1x-1x80xf16, -1x-1x1xf16)
        multiply_10 = paddle._C_ops.multiply(transpose_16, transpose_17)

        # pd_op.flatten: (-1x4x-1xf16, None) <- (-1x4x-1x-1xf16)
        flatten_28, flatten_29 = (lambda x, f: f(x))(paddle._C_ops.flatten(scale_4, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_18 = paddle._C_ops.transpose(flatten_28, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(scale__4, [1], full_int_array_168, full_int_array_169, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [1]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_18, [2], full_int_array_170, full_int_array_171, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_8 = paddle._C_ops.subtract(slice_42, slice_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(scale__4, [1], full_int_array_172, full_int_array_173, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [2]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_18, [2], full_int_array_174, full_int_array_175, [1], [2])

        # pd_op.subtract: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        subtract_9 = paddle._C_ops.subtract(slice_44, slice_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [1]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(scale__4, [1], full_int_array_176, full_int_array_177, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [3]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_18, [2], full_int_array_178, full_int_array_179, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_90 = paddle._C_ops.add(slice_46, slice_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [2]

        # pd_op.slice: (-1xf16) <- (-1x2xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(scale__4, [1], full_int_array_180, full_int_array_181, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [4]

        # pd_op.slice: (-1x-1xf16) <- (-1x-1x4xf16, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_18, [2], full_int_array_182, full_int_array_183, [1], [2])

        # pd_op.add: (-1x-1xf16) <- (-1xf16, -1x-1xf16)
        add_91 = paddle._C_ops.add(slice_48, slice_49)

        # builtin.combine: ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16]) <- (-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16)
        combine_19 = [subtract_8, subtract_9, add_90, add_91]

        # pd_op.stack: (-1x4x-1xf16) <- ([-1x-1xf16, -1x-1xf16, -1x-1xf16, -1x-1xf16])
        stack_9 = paddle._C_ops.stack(combine_19, 1)

        # pd_op.transpose: (-1x-1x4xf16) <- (-1x4x-1xf16)
        transpose_19 = paddle._C_ops.transpose(stack_9, [0, 2, 1])

        # builtin.combine: ([-1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16]) <- (-1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16)
        combine_20 = [transpose_3, transpose_7, transpose_11, transpose_15, transpose_19]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x-1x4xf16) <- ([-1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16, -1x-1x4xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_20, full_40)

        # builtin.combine: ([-1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16]) <- (-1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16)
        combine_21 = [multiply_6, multiply_7, multiply_8, multiply_9, multiply_10]

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x-1x80xf16) <- ([-1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16, -1x-1x80xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_21, full_41)

        # pd_op.cast: (-1x2xf16) <- (-1x2xf32)
        cast_16 = paddle._C_ops.cast(feed_1, paddle.float16)

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x1xf16, -1x1xf16]) <- (-1x2xf16, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(cast_16, 2, full_42)

        # builtin.slice: (-1x1xf16) <- ([-1x1xf16, -1x1xf16])
        slice_50 = split_with_num_0[1]

        # builtin.slice: (-1x1xf16) <- ([-1x1xf16, -1x1xf16])
        slice_51 = split_with_num_0[0]

        # builtin.combine: ([-1x1xf16, -1x1xf16, -1x1xf16, -1x1xf16]) <- (-1x1xf16, -1x1xf16, -1x1xf16, -1x1xf16)
        combine_22 = [slice_50, slice_51, slice_50, slice_51]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x4xf16) <- ([-1x1xf16, -1x1xf16, -1x1xf16, -1x1xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_22, full_43)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_184 = [-1, 1, 4]

        # pd_op.reshape_: (-1x1x4xf16, 0x-1x4xf16) <- (-1x4xf16, 3xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_2, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.divide: (-1x-1x4xf16) <- (-1x-1x4xf16, -1x1x4xf16)
        divide_0 = paddle._C_ops.divide(concat_0, reshape__20)

        # pd_op.transpose: (-1x80x-1xf16) <- (-1x-1x80xf16)
        transpose_20 = paddle._C_ops.transpose(concat_1, [0, 2, 1])

        # pd_op.cast: (-1x-1x4xf32) <- (-1x-1x4xf16)
        cast_17 = paddle._C_ops.cast(divide_0, paddle.float32)

        # pd_op.cast: (-1x80x-1xf32) <- (-1x80x-1xf16)
        cast_18 = paddle._C_ops.cast(transpose_20, paddle.float32)

        # pd_op.multiclass_nms3: (-1x6xf32, -1x1xi32, -1xi32) <- (-1x-1x4xf32, -1x80x-1xf32, None)
        multiclass_nms3_0, multiclass_nms3_1, multiclass_nms3_2 = (lambda x, f: f(x))(paddle._C_ops.multiclass_nms3(cast_17, cast_18, None, float('0.025'), 1000, 100, float('0.6'), True, float('1'), -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))
        return multiclass_nms3_0, multiclass_nms3_2



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_6, parameter_3, parameter_5, parameter_4, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_211, parameter_208, parameter_210, parameter_209, parameter_212, parameter_216, parameter_213, parameter_215, parameter_214, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_226, parameter_223, parameter_225, parameter_224, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_288, parameter_290, parameter_289, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_306, parameter_305, parameter_307, parameter_308, parameter_310, parameter_309, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, feed_1, feed_0):
        return self.builtin_module_1275_0_0(parameter_0, parameter_1, parameter_2, parameter_6, parameter_3, parameter_5, parameter_4, parameter_7, parameter_11, parameter_8, parameter_10, parameter_9, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_22, parameter_26, parameter_23, parameter_25, parameter_24, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_211, parameter_208, parameter_210, parameter_209, parameter_212, parameter_216, parameter_213, parameter_215, parameter_214, parameter_217, parameter_221, parameter_218, parameter_220, parameter_219, parameter_222, parameter_226, parameter_223, parameter_225, parameter_224, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_286, parameter_285, parameter_287, parameter_288, parameter_290, parameter_289, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_298, parameter_297, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_306, parameter_305, parameter_307, parameter_308, parameter_310, parameter_309, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, feed_1, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1275_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 3, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64, 3, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([2048, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([256, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_282
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([80, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([80], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([4, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([4], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([1, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1], dtype='float16', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1.0, 1.0], dtype='float32').reshape([1, 2]),
            # feed_0
            paddle.uniform([1, 3, 800, 1344], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
            # parameter_2
            paddle.static.InputSpec(shape=[64, 3, 7, 7], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[256, 2048, 1, 1], dtype='float16'),
            # parameter_272
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_282
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[80, 256, 3, 3], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[80], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[4, 256, 3, 3], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[4], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
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