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
    return [604][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_983_0_0(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_171, parameter_170, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_309, parameter_308, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_336, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_355, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0):

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf32)
        shape_0 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x3x224x224xf32, 32x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.gelu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        gelu_0 = paddle._C_ops.gelu(add__1, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x5x5xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(gelu_0, parameter_12, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__2 = paddle._C_ops.add_(depthwise_conv2d_0, reshape_4)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(add__2, parameter_14, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__3 = paddle._C_ops.add_(depthwise_conv2d_1, reshape_6)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(add__3, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_2, reshape_8)

        # pd_op.multiply_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        multiply__0 = paddle._C_ops.multiply_(gelu_0, add__4)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(multiply__0, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_3, reshape_10)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__6 = paddle._C_ops.add_(add__5, batch_norm__6)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(parameter_20, add__6)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__7 = paddle._C_ops.add_(batch_norm__0, multiply_0)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__7, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x32x56x56xf32, 256x32x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__12, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_26, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_4, reshape_12)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(add__8, parameter_27, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_28, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__9 = paddle._C_ops.add_(depthwise_conv2d_2, reshape_14)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_1 = paddle._C_ops.gelu(add__9, False)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x256x56x56xf32, 32x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(gelu_1, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_30, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_5, reshape_16)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_1 = paddle._C_ops.multiply(parameter_31, add__10)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__11 = paddle._C_ops.add_(add__7, multiply_1)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__11, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(batch_norm__18, parameter_36, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_37, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_6, reshape_18)

        # pd_op.gelu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        gelu_2 = paddle._C_ops.gelu(add__12, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(gelu_2, parameter_38, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_39, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__13 = paddle._C_ops.add_(depthwise_conv2d_3, reshape_20)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(add__13, parameter_40, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__14 = paddle._C_ops.add_(depthwise_conv2d_4, reshape_22)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(add__14, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_43, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_7, reshape_24)

        # pd_op.multiply_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        multiply__1 = paddle._C_ops.multiply_(gelu_2, add__15)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(multiply__1, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_45, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_8, reshape_26)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__17 = paddle._C_ops.add_(add__16, batch_norm__18)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_2 = paddle._C_ops.multiply(parameter_46, add__17)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__18 = paddle._C_ops.add_(add__11, multiply_2)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__18, parameter_47, parameter_48, parameter_49, parameter_50, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x32x56x56xf32, 256x32x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(batch_norm__24, parameter_51, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_52, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_9, reshape_28)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(add__19, parameter_53, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_54, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__20 = paddle._C_ops.add_(depthwise_conv2d_5, reshape_30)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_3 = paddle._C_ops.gelu(add__20, False)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x256x56x56xf32, 32x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(gelu_3, parameter_55, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_56, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_10, reshape_32)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_3 = paddle._C_ops.multiply(parameter_57, add__21)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__22 = paddle._C_ops.add_(add__18, multiply_3)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__22, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__30, parameter_62, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_63, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_11, reshape_34)

        # pd_op.gelu: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        gelu_4 = paddle._C_ops.gelu(add__23, False)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(gelu_4, parameter_64, [1, 1], [2, 2], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__24 = paddle._C_ops.add_(depthwise_conv2d_6, reshape_36)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(add__24, parameter_66, [1, 1], [9, 9], 'EXPLICIT', 32, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_67, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__25 = paddle._C_ops.add_(depthwise_conv2d_7, reshape_38)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(add__25, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_12, reshape_40)

        # pd_op.multiply_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        multiply__2 = paddle._C_ops.multiply_(gelu_4, add__26)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(multiply__2, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_71, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_13, reshape_42)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__28 = paddle._C_ops.add_(add__27, batch_norm__30)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_4 = paddle._C_ops.multiply(parameter_72, add__28)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__29 = paddle._C_ops.add_(add__22, multiply_4)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__29, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x32x56x56xf32, 256x32x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(batch_norm__36, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_78, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_14, reshape_44)

        # pd_op.depthwise_conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add__30, parameter_79, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_80, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 1x256x1x1xf32)
        add__31 = paddle._C_ops.add_(depthwise_conv2d_8, reshape_46)

        # pd_op.gelu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        gelu_5 = paddle._C_ops.gelu(add__31, False)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x256x56x56xf32, 32x256x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(gelu_5, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_82, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_15, reshape_48)

        # pd_op.multiply: (-1x32x56x56xf32) <- (32x1x1xf32, -1x32x56x56xf32)
        multiply_5 = paddle._C_ops.multiply(parameter_83, add__32)

        # pd_op.add_: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add__33 = paddle._C_ops.add_(add__29, multiply_5)

        # pd_op.flatten_: (-1x32x3136xf32, None) <- (-1x32x56x56xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__33, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x32xf32) <- (-1x32x3136xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x32xf32, -3136xf32, -3136xf32) <- (-1x3136x32xf32, 32xf32, 32xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_84, parameter_85, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1, full_2]

        # pd_op.reshape_: (-1x56x56x32xf32, 0x-1x3136x32xf32) <- (-1x3136x32xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_0, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x32x56x56xf32) <- (-1x56x56x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(transpose_1, parameter_86, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_87, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_16, reshape_50)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__34, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__42, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__48, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_97, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_17, reshape_52)

        # pd_op.gelu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        gelu_6 = paddle._C_ops.gelu(add__35, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(gelu_6, parameter_98, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_99, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__36 = paddle._C_ops.add_(depthwise_conv2d_9, reshape_54)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(add__36, parameter_100, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__37 = paddle._C_ops.add_(depthwise_conv2d_10, reshape_56)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(add__37, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_18, reshape_58)

        # pd_op.multiply_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        multiply__3 = paddle._C_ops.multiply_(gelu_6, add__38)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(multiply__3, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_19, reshape_60)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__40 = paddle._C_ops.add_(add__39, batch_norm__48)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_6 = paddle._C_ops.multiply(parameter_106, add__40)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__41 = paddle._C_ops.add_(batch_norm__42, multiply_6)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__41, parameter_107, parameter_108, parameter_109, parameter_110, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x64x28x28xf32, 512x64x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(batch_norm__54, parameter_111, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_112, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__42 = paddle._C_ops.add_(conv2d_20, reshape_62)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(add__42, parameter_113, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_114, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__43 = paddle._C_ops.add_(depthwise_conv2d_11, reshape_64)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_7 = paddle._C_ops.gelu(add__43, False)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x512x28x28xf32, 64x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(gelu_7, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_116, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_21, reshape_66)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_7 = paddle._C_ops.multiply(parameter_117, add__44)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__45 = paddle._C_ops.add_(add__41, multiply_7)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__45, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(batch_norm__60, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_123, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_22, reshape_68)

        # pd_op.gelu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        gelu_8 = paddle._C_ops.gelu(add__46, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(gelu_8, parameter_124, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_125, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__47 = paddle._C_ops.add_(depthwise_conv2d_12, reshape_70)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add__47, parameter_126, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_127, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__48 = paddle._C_ops.add_(depthwise_conv2d_13, reshape_72)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(add__48, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_129, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__49 = paddle._C_ops.add_(conv2d_23, reshape_74)

        # pd_op.multiply_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        multiply__4 = paddle._C_ops.multiply_(gelu_8, add__49)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(multiply__4, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_131, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__50 = paddle._C_ops.add_(conv2d_24, reshape_76)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__51 = paddle._C_ops.add_(add__50, batch_norm__60)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_8 = paddle._C_ops.multiply(parameter_132, add__51)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__52 = paddle._C_ops.add_(add__45, multiply_8)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__52, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x64x28x28xf32, 512x64x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(batch_norm__66, parameter_137, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_138, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__53 = paddle._C_ops.add_(conv2d_25, reshape_78)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(add__53, parameter_139, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_140, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__54 = paddle._C_ops.add_(depthwise_conv2d_14, reshape_80)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_9 = paddle._C_ops.gelu(add__54, False)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x512x28x28xf32, 64x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(gelu_9, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_142, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__55 = paddle._C_ops.add_(conv2d_26, reshape_82)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_9 = paddle._C_ops.multiply(parameter_143, add__55)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__56 = paddle._C_ops.add_(add__52, multiply_9)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__56, parameter_144, parameter_145, parameter_146, parameter_147, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(batch_norm__72, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_149, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__57 = paddle._C_ops.add_(conv2d_27, reshape_84)

        # pd_op.gelu: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        gelu_10 = paddle._C_ops.gelu(add__57, False)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x5x5xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(gelu_10, parameter_150, [1, 1], [2, 2], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__58 = paddle._C_ops.add_(depthwise_conv2d_15, reshape_86)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(add__58, parameter_152, [1, 1], [9, 9], 'EXPLICIT', 64, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_153, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__59 = paddle._C_ops.add_(depthwise_conv2d_16, reshape_88)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(add__59, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_155, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__60 = paddle._C_ops.add_(conv2d_28, reshape_90)

        # pd_op.multiply_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        multiply__5 = paddle._C_ops.multiply_(gelu_10, add__60)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(multiply__5, parameter_156, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_157, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__61 = paddle._C_ops.add_(conv2d_29, reshape_92)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__62 = paddle._C_ops.add_(add__61, batch_norm__72)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_10 = paddle._C_ops.multiply(parameter_158, add__62)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__63 = paddle._C_ops.add_(add__56, multiply_10)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__63, parameter_159, parameter_160, parameter_161, parameter_162, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x64x28x28xf32, 512x64x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(batch_norm__78, parameter_163, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_164, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__64 = paddle._C_ops.add_(conv2d_30, reshape_94)

        # pd_op.depthwise_conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x1x3x3xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(add__64, parameter_165, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_166, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 1x512x1x1xf32)
        add__65 = paddle._C_ops.add_(depthwise_conv2d_17, reshape_96)

        # pd_op.gelu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        gelu_11 = paddle._C_ops.gelu(add__65, False)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x512x28x28xf32, 64x512x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(gelu_11, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_168, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add__66 = paddle._C_ops.add_(conv2d_31, reshape_98)

        # pd_op.multiply: (-1x64x28x28xf32) <- (64x1x1xf32, -1x64x28x28xf32)
        multiply_11 = paddle._C_ops.multiply(parameter_169, add__66)

        # pd_op.add_: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add__67 = paddle._C_ops.add_(add__63, multiply_11)

        # pd_op.flatten_: (-1x64x784xf32, None) <- (-1x64x28x28xf32)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__67, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x64xf32) <- (-1x64x784xf32)
        transpose_2 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x64xf32, -784xf32, -784xf32) <- (-1x784x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_2, parameter_170, parameter_171, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, full_3, full_4, full_5]

        # pd_op.reshape_: (-1x28x28x64xf32, 0x-1x784x64xf32) <- (-1x784x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_3, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x28x28xf32) <- (-1x28x28x64xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__2, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x64x28x28xf32, 160x64x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(transpose_3, parameter_172, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_173, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__68 = paddle._C_ops.add_(conv2d_32, reshape_100)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__68, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__84, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(batch_norm__90, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__69 = paddle._C_ops.add_(conv2d_33, reshape_102)

        # pd_op.gelu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        gelu_12 = paddle._C_ops.gelu(add__69, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x5x5xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(gelu_12, parameter_184, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__70 = paddle._C_ops.add_(depthwise_conv2d_18, reshape_104)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x7x7xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(add__70, parameter_186, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_187, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__71 = paddle._C_ops.add_(depthwise_conv2d_19, reshape_106)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(add__71, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__72 = paddle._C_ops.add_(conv2d_34, reshape_108)

        # pd_op.multiply_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        multiply__6 = paddle._C_ops.multiply_(gelu_12, add__72)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(multiply__6, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__73 = paddle._C_ops.add_(conv2d_35, reshape_110)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__74 = paddle._C_ops.add_(add__73, batch_norm__90)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_12 = paddle._C_ops.multiply(parameter_192, add__74)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__75 = paddle._C_ops.add_(batch_norm__84, multiply_12)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__75, parameter_193, parameter_194, parameter_195, parameter_196, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x160x14x14xf32, 640x160x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(batch_norm__96, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_198, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__76 = paddle._C_ops.add_(conv2d_36, reshape_112)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(add__76, parameter_199, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_200, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__77 = paddle._C_ops.add_(depthwise_conv2d_20, reshape_114)

        # pd_op.gelu: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        gelu_13 = paddle._C_ops.gelu(add__77, False)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x640x14x14xf32, 160x640x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(gelu_13, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_202, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__78 = paddle._C_ops.add_(conv2d_37, reshape_116)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_13 = paddle._C_ops.multiply(parameter_203, add__78)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__79 = paddle._C_ops.add_(add__75, multiply_13)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__79, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__102, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_209, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__80 = paddle._C_ops.add_(conv2d_38, reshape_118)

        # pd_op.gelu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        gelu_14 = paddle._C_ops.gelu(add__80, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x5x5xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(gelu_14, parameter_210, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_211, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__81 = paddle._C_ops.add_(depthwise_conv2d_21, reshape_120)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x7x7xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(add__81, parameter_212, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_213, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__82 = paddle._C_ops.add_(depthwise_conv2d_22, reshape_122)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(add__82, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_64 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__83 = paddle._C_ops.add_(conv2d_39, reshape_124)

        # pd_op.multiply_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        multiply__7 = paddle._C_ops.multiply_(gelu_14, add__83)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(multiply__7, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_126, reshape_127 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_217, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__84 = paddle._C_ops.add_(conv2d_40, reshape_126)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__85 = paddle._C_ops.add_(add__84, batch_norm__102)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_14 = paddle._C_ops.multiply(parameter_218, add__85)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__86 = paddle._C_ops.add_(add__79, multiply_14)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__86, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x160x14x14xf32, 640x160x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(batch_norm__108, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_128, reshape_129 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_224, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__87 = paddle._C_ops.add_(conv2d_41, reshape_128)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(add__87, parameter_225, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_130, reshape_131 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_226, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__88 = paddle._C_ops.add_(depthwise_conv2d_23, reshape_130)

        # pd_op.gelu: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        gelu_15 = paddle._C_ops.gelu(add__88, False)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x640x14x14xf32, 160x640x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(gelu_15, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_132, reshape_133 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_228, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__89 = paddle._C_ops.add_(conv2d_42, reshape_132)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_15 = paddle._C_ops.multiply(parameter_229, add__89)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__90 = paddle._C_ops.add_(add__86, multiply_15)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__90, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(batch_norm__114, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_134, reshape_135 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_235, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__91 = paddle._C_ops.add_(conv2d_43, reshape_134)

        # pd_op.gelu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        gelu_16 = paddle._C_ops.gelu(add__91, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x5x5xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(gelu_16, parameter_236, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_70 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_136, reshape_137 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_237, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__92 = paddle._C_ops.add_(depthwise_conv2d_24, reshape_136)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x7x7xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(add__92, parameter_238, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_71 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_138, reshape_139 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_239, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__93 = paddle._C_ops.add_(depthwise_conv2d_25, reshape_138)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(add__93, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_72 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_140, reshape_141 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_241, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__94 = paddle._C_ops.add_(conv2d_44, reshape_140)

        # pd_op.multiply_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        multiply__8 = paddle._C_ops.multiply_(gelu_16, add__94)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(multiply__8, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_73 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_142, reshape_143 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_243, full_int_array_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__95 = paddle._C_ops.add_(conv2d_45, reshape_142)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__96 = paddle._C_ops.add_(add__95, batch_norm__114)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_16 = paddle._C_ops.multiply(parameter_244, add__96)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__97 = paddle._C_ops.add_(add__90, multiply_16)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__97, parameter_245, parameter_246, parameter_247, parameter_248, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x160x14x14xf32, 640x160x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(batch_norm__120, parameter_249, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_74 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_144, reshape_145 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_250, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__98 = paddle._C_ops.add_(conv2d_46, reshape_144)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(add__98, parameter_251, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_75 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_146, reshape_147 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_252, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__99 = paddle._C_ops.add_(depthwise_conv2d_26, reshape_146)

        # pd_op.gelu: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        gelu_17 = paddle._C_ops.gelu(add__99, False)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x640x14x14xf32, 160x640x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(gelu_17, parameter_253, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_76 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_148, reshape_149 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_254, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__100 = paddle._C_ops.add_(conv2d_47, reshape_148)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_17 = paddle._C_ops.multiply(parameter_255, add__100)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__101 = paddle._C_ops.add_(add__97, multiply_17)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__101, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(batch_norm__126, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_77 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_150, reshape_151 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_261, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__102 = paddle._C_ops.add_(conv2d_48, reshape_150)

        # pd_op.gelu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        gelu_18 = paddle._C_ops.gelu(add__102, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x5x5xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(gelu_18, parameter_262, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_78 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_152, reshape_153 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_263, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__103 = paddle._C_ops.add_(depthwise_conv2d_27, reshape_152)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x7x7xf32)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(add__103, parameter_264, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_154, reshape_155 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_265, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__104 = paddle._C_ops.add_(depthwise_conv2d_28, reshape_154)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(add__104, parameter_266, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_156, reshape_157 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_267, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__105 = paddle._C_ops.add_(conv2d_49, reshape_156)

        # pd_op.multiply_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        multiply__9 = paddle._C_ops.multiply_(gelu_18, add__105)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(multiply__9, parameter_268, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_81 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_158, reshape_159 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_269, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__106 = paddle._C_ops.add_(conv2d_50, reshape_158)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__107 = paddle._C_ops.add_(add__106, batch_norm__126)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_18 = paddle._C_ops.multiply(parameter_270, add__107)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__108 = paddle._C_ops.add_(add__101, multiply_18)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__108, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x160x14x14xf32, 640x160x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(batch_norm__132, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_82 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_160, reshape_161 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_276, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__109 = paddle._C_ops.add_(conv2d_51, reshape_160)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(add__109, parameter_277, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_83 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_162, reshape_163 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_278, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__110 = paddle._C_ops.add_(depthwise_conv2d_29, reshape_162)

        # pd_op.gelu: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        gelu_19 = paddle._C_ops.gelu(add__110, False)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x640x14x14xf32, 160x640x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(gelu_19, parameter_279, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_84 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_164, reshape_165 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_280, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__111 = paddle._C_ops.add_(conv2d_52, reshape_164)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_19 = paddle._C_ops.multiply(parameter_281, add__111)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__112 = paddle._C_ops.add_(add__108, multiply_19)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__112, parameter_282, parameter_283, parameter_284, parameter_285, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(batch_norm__138, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_166, reshape_167 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_287, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__113 = paddle._C_ops.add_(conv2d_53, reshape_166)

        # pd_op.gelu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        gelu_20 = paddle._C_ops.gelu(add__113, False)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x5x5xf32)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(gelu_20, parameter_288, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_168, reshape_169 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_289, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__114 = paddle._C_ops.add_(depthwise_conv2d_30, reshape_168)

        # pd_op.depthwise_conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x1x7x7xf32)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(add__114, parameter_290, [1, 1], [9, 9], 'EXPLICIT', 160, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_170, reshape_171 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_291, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__115 = paddle._C_ops.add_(depthwise_conv2d_31, reshape_170)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(add__115, parameter_292, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_88 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_172, reshape_173 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_293, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__116 = paddle._C_ops.add_(conv2d_54, reshape_172)

        # pd_op.multiply_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        multiply__10 = paddle._C_ops.multiply_(gelu_20, add__116)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(multiply__10, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_89 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_174, reshape_175 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_295, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__117 = paddle._C_ops.add_(conv2d_55, reshape_174)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__118 = paddle._C_ops.add_(add__117, batch_norm__138)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_20 = paddle._C_ops.multiply(parameter_296, add__118)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__119 = paddle._C_ops.add_(add__112, multiply_20)

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__119, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x160x14x14xf32, 640x160x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(batch_norm__144, parameter_301, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_90 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_176, reshape_177 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_302, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__120 = paddle._C_ops.add_(conv2d_56, reshape_176)

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x3x3xf32)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(add__120, parameter_303, [1, 1], [1, 1], 'EXPLICIT', 640, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_91 = [1, 640, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32, 0x640xf32) <- (640xf32, 4xi64)
        reshape_178, reshape_179 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 1x640x1x1xf32)
        add__121 = paddle._C_ops.add_(depthwise_conv2d_32, reshape_178)

        # pd_op.gelu: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        gelu_21 = paddle._C_ops.gelu(add__121, False)

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x640x14x14xf32, 160x640x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(gelu_21, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_92 = [1, 160, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32, 0x160xf32) <- (160xf32, 4xi64)
        reshape_180, reshape_181 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_306, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 1x160x1x1xf32)
        add__122 = paddle._C_ops.add_(conv2d_57, reshape_180)

        # pd_op.multiply: (-1x160x14x14xf32) <- (160x1x1xf32, -1x160x14x14xf32)
        multiply_21 = paddle._C_ops.multiply(parameter_307, add__122)

        # pd_op.add_: (-1x160x14x14xf32) <- (-1x160x14x14xf32, -1x160x14x14xf32)
        add__123 = paddle._C_ops.add_(add__119, multiply_21)

        # pd_op.flatten_: (-1x160x196xf32, None) <- (-1x160x14x14xf32)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__123, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x160xf32) <- (-1x160x196xf32)
        transpose_4 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x160xf32, -196xf32, -196xf32) <- (-1x196x160xf32, 160xf32, 160xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_4, parameter_308, parameter_309, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_0, full_6, full_7, full_8]

        # pd_op.reshape_: (-1x14x14x160xf32, 0x-1x196x160xf32) <- (-1x196x160xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_6, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x14x14xf32) <- (-1x14x14x160xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__4, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x160x14x14xf32, 256x160x3x3xf32)
        conv2d_58 = paddle._C_ops.conv2d(transpose_5, parameter_310, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_93 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_182, reshape_183 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_311, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__124 = paddle._C_ops.add_(conv2d_58, reshape_182)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__124, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(batch_norm__150, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(batch_norm__156, parameter_320, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_94 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_184, reshape_185 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__125 = paddle._C_ops.add_(conv2d_59, reshape_184)

        # pd_op.gelu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        gelu_22 = paddle._C_ops.gelu(add__125, False)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(gelu_22, parameter_322, [1, 1], [2, 2], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_95 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_186, reshape_187 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_323, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__126 = paddle._C_ops.add_(depthwise_conv2d_33, reshape_186)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(add__126, parameter_324, [1, 1], [9, 9], 'EXPLICIT', 256, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_96 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_188, reshape_189 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_325, full_int_array_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__127 = paddle._C_ops.add_(depthwise_conv2d_34, reshape_188)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(add__127, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_97 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_190, reshape_191 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_327, full_int_array_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__128 = paddle._C_ops.add_(conv2d_60, reshape_190)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        multiply__11 = paddle._C_ops.multiply_(gelu_22, add__128)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(multiply__11, parameter_328, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_98 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_192, reshape_193 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_329, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__129 = paddle._C_ops.add_(conv2d_61, reshape_192)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__130 = paddle._C_ops.add_(add__129, batch_norm__156)

        # pd_op.multiply: (-1x256x7x7xf32) <- (256x1x1xf32, -1x256x7x7xf32)
        multiply_22 = paddle._C_ops.multiply(parameter_330, add__130)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__131 = paddle._C_ops.add_(batch_norm__150, multiply_22)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__131, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(batch_norm__162, parameter_335, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_194, reshape_195 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_336, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__132 = paddle._C_ops.add_(conv2d_62, reshape_194)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x1x3x3xf32)
        depthwise_conv2d_35 = paddle._C_ops.depthwise_conv2d(add__132, parameter_337, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_100 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_196, reshape_197 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_338, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__133 = paddle._C_ops.add_(depthwise_conv2d_35, reshape_196)

        # pd_op.gelu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        gelu_23 = paddle._C_ops.gelu(add__133, False)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(gelu_23, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_101 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_198, reshape_199 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_340, full_int_array_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__134 = paddle._C_ops.add_(conv2d_63, reshape_198)

        # pd_op.multiply: (-1x256x7x7xf32) <- (256x1x1xf32, -1x256x7x7xf32)
        multiply_23 = paddle._C_ops.multiply(parameter_341, add__134)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__135 = paddle._C_ops.add_(add__131, multiply_23)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__135, parameter_342, parameter_343, parameter_344, parameter_345, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(batch_norm__168, parameter_346, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_102 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_200, reshape_201 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_347, full_int_array_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__136 = paddle._C_ops.add_(conv2d_64, reshape_200)

        # pd_op.gelu: (-1x256x7x7xf32) <- (-1x256x7x7xf32)
        gelu_24 = paddle._C_ops.gelu(add__136, False)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x5x5xf32)
        depthwise_conv2d_36 = paddle._C_ops.depthwise_conv2d(gelu_24, parameter_348, [1, 1], [2, 2], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_103 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_202, reshape_203 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_349, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__137 = paddle._C_ops.add_(depthwise_conv2d_36, reshape_202)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_37 = paddle._C_ops.depthwise_conv2d(add__137, parameter_350, [1, 1], [9, 9], 'EXPLICIT', 256, [3, 3], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_104 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_204, reshape_205 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_351, full_int_array_104), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__138 = paddle._C_ops.add_(depthwise_conv2d_37, reshape_204)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(add__138, parameter_352, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_105 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_206, reshape_207 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_353, full_int_array_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__139 = paddle._C_ops.add_(conv2d_65, reshape_206)

        # pd_op.multiply_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        multiply__12 = paddle._C_ops.multiply_(gelu_24, add__139)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(multiply__12, parameter_354, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_106 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_208, reshape_209 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_355, full_int_array_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__140 = paddle._C_ops.add_(conv2d_66, reshape_208)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__141 = paddle._C_ops.add_(add__140, batch_norm__168)

        # pd_op.multiply: (-1x256x7x7xf32) <- (256x1x1xf32, -1x256x7x7xf32)
        multiply_24 = paddle._C_ops.multiply(parameter_356, add__141)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__142 = paddle._C_ops.add_(add__135, multiply_24)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__142, parameter_357, parameter_358, parameter_359, parameter_360, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(batch_norm__174, parameter_361, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_107 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_210, reshape_211 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_362, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__143 = paddle._C_ops.add_(conv2d_67, reshape_210)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x1x3x3xf32)
        depthwise_conv2d_38 = paddle._C_ops.depthwise_conv2d(add__143, parameter_363, [1, 1], [1, 1], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_108 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_212, reshape_213 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_364, full_int_array_108), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add__144 = paddle._C_ops.add_(depthwise_conv2d_38, reshape_212)

        # pd_op.gelu: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        gelu_25 = paddle._C_ops.gelu(add__144, False)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(gelu_25, parameter_365, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_109 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_214, reshape_215 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_366, full_int_array_109), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add__145 = paddle._C_ops.add_(conv2d_68, reshape_214)

        # pd_op.multiply: (-1x256x7x7xf32) <- (256x1x1xf32, -1x256x7x7xf32)
        multiply_25 = paddle._C_ops.multiply(parameter_367, add__145)

        # pd_op.add_: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add__146 = paddle._C_ops.add_(add__142, multiply_25)

        # pd_op.flatten_: (-1x256x49xf32, None) <- (-1x256x7x7xf32)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__146, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x256xf32) <- (-1x256x49xf32)
        transpose_6 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x256xf32, -49xf32, -49xf32) <- (-1x49x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_6, parameter_368, parameter_369, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x256xf32) <- (-1x49x256xf32)
        mean_0 = paddle._C_ops.mean(layer_norm_9, [1], False)

        # pd_op.matmul: (-1x1000xf32) <- (-1x256xf32, 256x1000xf32)
        matmul_0 = paddle._C_ops.matmul(mean_0, parameter_370, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__147 = paddle._C_ops.add_(matmul_0, parameter_371)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax_(add__147, -1)
        return softmax__0



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

    def forward(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_171, parameter_170, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_309, parameter_308, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_336, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_355, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0):
        return self.builtin_module_983_0_0(parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_50, parameter_47, parameter_49, parameter_48, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_147, parameter_144, parameter_146, parameter_145, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_171, parameter_170, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_250, parameter_251, parameter_252, parameter_253, parameter_254, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_268, parameter_269, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_294, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_309, parameter_308, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_336, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_347, parameter_348, parameter_349, parameter_350, parameter_351, parameter_352, parameter_353, parameter_354, parameter_355, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_363, parameter_364, parameter_365, parameter_366, parameter_367, parameter_369, parameter_368, parameter_370, parameter_371, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_983_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
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
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([32, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([32, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([32, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([64, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([64, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([160, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([160, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([160, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([160, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([160, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([640, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([160, 640, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([256, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([256, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([256, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([256, 1, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([1024, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([256, 1000], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 7, 7], dtype='float32'),
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
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[32, 1, 5, 5], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[32, 1, 7, 7], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256, 32, 1, 1], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[32, 256, 1, 1], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[64, 1, 5, 5], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[64, 1, 7, 7], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512, 64, 1, 1], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[64, 512, 1, 1], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[160, 64, 3, 3], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[640, 1, 3, 3], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[160, 640, 1, 1], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[256, 160, 3, 3], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[256, 1, 5, 5], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[256, 1, 7, 7], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[256, 1, 5, 5], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[256, 1, 7, 7], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[1024, 1, 3, 3], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[256, 1000], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
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