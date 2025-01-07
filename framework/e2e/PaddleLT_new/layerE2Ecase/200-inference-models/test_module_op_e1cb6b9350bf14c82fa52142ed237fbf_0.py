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
    return [875][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1463_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_356, parameter_353, parameter_355, parameter_354, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_406, parameter_403, parameter_405, parameter_404, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_416, parameter_413, parameter_415, parameter_414, parameter_417, parameter_421, parameter_418, parameter_420, parameter_419, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_431, parameter_428, parameter_430, parameter_429, parameter_432, parameter_436, parameter_433, parameter_435, parameter_434, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_456, parameter_453, parameter_455, parameter_454, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_531, parameter_532, parameter_536, parameter_533, parameter_535, parameter_534, parameter_537, parameter_541, parameter_538, parameter_540, parameter_539, parameter_542, parameter_546, parameter_543, parameter_545, parameter_544, parameter_547, parameter_551, parameter_548, parameter_550, parameter_549, parameter_552, parameter_553, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_575, parameter_576, parameter_577, feed_1, feed_0):

        # pd_op.conv2d: (-1x24x160x160xf32) <- (-1x3x320x320xf32, 24x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x160x160xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x24x160x160xf32) <- (-1x24x160x160xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x24x80x80xf32) <- (-1x24x160x160xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(hardswish_0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.depthwise_conv2d: (-1x24x40x40xf32) <- (-1x24x80x80xf32, 24x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(pool2d_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x40x40xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x40x40xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x24x40x40xf32, 48x24x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (-1x44x80x80xf32) <- (-1x24x80x80xf32, 44x24x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x44x80x80xf32, 44xf32, 44xf32, xf32, xf32, None) <- (-1x44x80x80xf32, 44xf32, 44xf32, 44xf32, 44xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x44x80x80xf32) <- (-1x44x80x80xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__18)

        # pd_op.depthwise_conv2d: (-1x44x40x40xf32) <- (-1x44x80x80xf32, 44x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(hardswish_2, parameter_20, [2, 2], [1, 1], 'EXPLICIT', 44, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x44x40x40xf32, 44xf32, 44xf32, xf32, xf32, None) <- (-1x44x40x40xf32, 44xf32, 44xf32, 44xf32, 44xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x44x1x1xf32) <- (-1x44x40x40xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(batch_norm__24, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x11x1x1xf32) <- (-1x44x1x1xf32, 11x44x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 11, 1, 1]

        # pd_op.reshape: (1x11x1x1xf32, 0x11xf32) <- (11xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_26, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x11x1x1xf32) <- (-1x11x1x1xf32, 1x11x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_3, reshape_0)

        # pd_op.relu_: (-1x11x1x1xf32) <- (-1x11x1x1xf32)
        relu__0 = paddle._C_ops.relu_(add__0)

        # pd_op.conv2d: (-1x44x1x1xf32) <- (-1x11x1x1xf32, 44x11x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__0, parameter_27, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 44, 1, 1]

        # pd_op.reshape: (1x44x1x1xf32, 0x44xf32) <- (44xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_28, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x44x1x1xf32) <- (-1x44x1x1xf32, 1x44x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_4, reshape_2)

        # pd_op.hardsigmoid: (-1x44x1x1xf32) <- (-1x44x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add__1, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x44x40x40xf32) <- (-1x44x40x40xf32, -1x44x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(batch_norm__24, hardsigmoid_0)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x44x40x40xf32, 48x44x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(multiply__0, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__30)

        # builtin.combine: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x48x40x40xf32, -1x48x40x40xf32)
        combine_0 = [hardswish_1, hardswish_3]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.depthwise_conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(concat_0, parameter_34, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_35, parameter_36, parameter_37, parameter_38, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__36)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(hardswish_4, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__42)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [48, 48]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x96x40x40xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(hardswish_5, full_int_array_4, full_1)

        # builtin.slice: (-1x48x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32])
        slice_0 = split_0[1]

        # pd_op.conv2d: (-1x24x40x40xf32) <- (-1x48x40x40xf32, 24x48x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(slice_0, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x40x40xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x40x40xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x24x40x40xf32) <- (-1x24x40x40xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__48)

        # pd_op.depthwise_conv2d: (-1x24x40x40xf32) <- (-1x24x40x40xf32, 24x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(hardswish_6, parameter_49, [1, 1], [1, 1], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x40x40xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x40x40xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x24x40x40xf32, -1x24x40x40xf32]) <- (-1x24x40x40xf32, -1x24x40x40xf32)
        combine_1 = [hardswish_6, batch_norm__54]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x48x40x40xf32) <- ([-1x24x40x40xf32, -1x24x40x40xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x48x1x1xf32) <- (-1x48x40x40xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(concat_1, full_int_array_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x12x1x1xf32) <- (-1x48x1x1xf32, 12x48x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_2, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 12, 1, 1]

        # pd_op.reshape: (1x12x1x1xf32, 0x12xf32) <- (12xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x12x1x1xf32) <- (-1x12x1x1xf32, 1x12x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_8, reshape_4)

        # pd_op.relu_: (-1x12x1x1xf32) <- (-1x12x1x1xf32)
        relu__1 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x12x1x1xf32, 48x12x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__1, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_9, reshape_6)

        # pd_op.hardsigmoid: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add__3, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x48x40x40xf32) <- (-1x48x40x40xf32, -1x48x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(concat_1, hardsigmoid_1)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x48x40x40xf32, 48x48x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(multiply__1, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_59, parameter_60, parameter_61, parameter_62, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__60)

        # builtin.slice: (-1x48x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32])
        slice_1 = split_0[0]

        # builtin.combine: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x48x40x40xf32, -1x48x40x40xf32)
        combine_2 = [slice_1, hardswish_7]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_3)

        # pd_op.shape: (4xi32) <- (-1x96x40x40xf32)
        shape_0 = paddle._C_ops.shape(concat_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_2, full_4, full_5, full_6, full_7]

        # pd_op.reshape_: (-1x2x48x40x40xf32, 0x-1x96x40x40xf32) <- (-1x96x40x40xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_2, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x48x2x40x40xf32) <- (-1x2x48x40x40xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_2, full_8, full_9, full_10]

        # pd_op.reshape_: (-1x96x40x40xf32, 0x-1x48x2x40x40xf32) <- (-1x48x2x40x40xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [48, 48]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x96x40x40xf32, 2xi64, 1xi32)
        split_1 = paddle._C_ops.split(reshape__2, full_int_array_10, full_11)

        # builtin.slice: (-1x48x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32])
        slice_3 = split_1[1]

        # pd_op.conv2d: (-1x24x40x40xf32) <- (-1x48x40x40xf32, 24x48x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(slice_3, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x40x40xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x40x40xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x24x40x40xf32) <- (-1x24x40x40xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__66)

        # pd_op.depthwise_conv2d: (-1x24x40x40xf32) <- (-1x24x40x40xf32, 24x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(hardswish_8, parameter_68, [1, 1], [1, 1], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x40x40xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x40x40xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_69, parameter_70, parameter_71, parameter_72, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x24x40x40xf32, -1x24x40x40xf32]) <- (-1x24x40x40xf32, -1x24x40x40xf32)
        combine_5 = [hardswish_8, batch_norm__72]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x48x40x40xf32) <- ([-1x24x40x40xf32, -1x24x40x40xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_5, full_12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [1, 1]

        # pd_op.pool2d: (-1x48x1x1xf32) <- (-1x48x40x40xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(concat_3, full_int_array_11, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x12x1x1xf32) <- (-1x48x1x1xf32, 12x48x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(pool2d_3, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 12, 1, 1]

        # pd_op.reshape: (1x12x1x1xf32, 0x12xf32) <- (12xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_74, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x12x1x1xf32) <- (-1x12x1x1xf32, 1x12x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_12, reshape_8)

        # pd_op.relu_: (-1x12x1x1xf32) <- (-1x12x1x1xf32)
        relu__2 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x12x1x1xf32, 48x12x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu__2, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_76, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_13, reshape_10)

        # pd_op.hardsigmoid: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add__5, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x48x40x40xf32) <- (-1x48x40x40xf32, -1x48x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(concat_3, hardsigmoid_2)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x48x40x40xf32, 48x48x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(multiply__2, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__78)

        # builtin.slice: (-1x48x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32])
        slice_4 = split_1[0]

        # builtin.combine: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x48x40x40xf32, -1x48x40x40xf32)
        combine_6 = [slice_4, hardswish_9]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_6, full_13)

        # pd_op.shape: (4xi32) <- (-1x96x40x40xf32)
        shape_1 = paddle._C_ops.shape(concat_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, full_14, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x2x48x40x40xf32, 0x-1x96x40x40xf32) <- (-1x96x40x40xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_4, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x48x2x40x40xf32) <- (-1x2x48x40x40xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_5, full_18, full_19, full_20]

        # pd_op.reshape_: (-1x96x40x40xf32, 0x-1x48x2x40x40xf32) <- (-1x48x2x40x40xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x96x20x20xf32) <- (-1x96x40x40xf32, 96x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(reshape__6, parameter_82, [2, 2], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_83, parameter_84, parameter_85, parameter_86, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__84, parameter_87, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__90)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x96x40x40xf32, 48x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(reshape__6, parameter_92, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_93, parameter_94, parameter_95, parameter_96, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__96)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x40x40xf32, 48x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(hardswish_11, parameter_97, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_98, parameter_99, parameter_100, parameter_101, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [1, 1]

        # pd_op.pool2d: (-1x48x1x1xf32) <- (-1x48x20x20xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(batch_norm__102, full_int_array_16, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x12x1x1xf32) <- (-1x48x1x1xf32, 12x48x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(pool2d_4, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 12, 1, 1]

        # pd_op.reshape: (1x12x1x1xf32, 0x12xf32) <- (12xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x12x1x1xf32) <- (-1x12x1x1xf32, 1x12x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_17, reshape_12)

        # pd_op.relu_: (-1x12x1x1xf32) <- (-1x12x1x1xf32)
        relu__3 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x12x1x1xf32, 48x12x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__3, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_18, reshape_14)

        # pd_op.hardsigmoid: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add__7, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x48x20x20xf32) <- (-1x48x20x20xf32, -1x48x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(batch_norm__102, hardsigmoid_3)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x48x20x20xf32, 96x48x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(multiply__3, parameter_106, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_107, parameter_108, parameter_109, parameter_110, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__108)

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_9 = [hardswish_10, hardswish_12]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_21)

        # pd_op.depthwise_conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(concat_5, parameter_111, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_112, parameter_113, parameter_114, parameter_115, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(hardswish_13, parameter_116, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_117, parameter_118, parameter_119, parameter_120, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(hardswish_14, full_int_array_19, full_22)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_6 = split_2[1]

        # pd_op.conv2d: (-1x60x20x20xf32) <- (-1x96x20x20xf32, 60x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(slice_6, parameter_121, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x20x20xf32, 60xf32, 60xf32, xf32, xf32, None) <- (-1x60x20x20xf32, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_122, parameter_123, parameter_124, parameter_125, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x20x20xf32) <- (-1x60x20x20xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__126)

        # pd_op.depthwise_conv2d: (-1x60x20x20xf32) <- (-1x60x20x20xf32, 60x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(hardswish_15, parameter_126, [1, 1], [1, 1], 'EXPLICIT', 60, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x60x20x20xf32, 60xf32, 60xf32, xf32, xf32, None) <- (-1x60x20x20xf32, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x60x20x20xf32, -1x60x20x20xf32]) <- (-1x60x20x20xf32, -1x60x20x20xf32)
        combine_10 = [hardswish_15, batch_norm__132]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x20x20xf32) <- ([-1x60x20x20xf32, -1x60x20x20xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_10, full_23)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [1, 1]

        # pd_op.pool2d: (-1x120x1x1xf32) <- (-1x120x20x20xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(concat_6, full_int_array_20, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x30x1x1xf32) <- (-1x120x1x1xf32, 30x120x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_5, parameter_131, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 30, 1, 1]

        # pd_op.reshape: (1x30x1x1xf32, 0x30xf32) <- (30xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_132, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x30x1x1xf32) <- (-1x30x1x1xf32, 1x30x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_22, reshape_16)

        # pd_op.relu_: (-1x30x1x1xf32) <- (-1x30x1x1xf32)
        relu__4 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x120x1x1xf32) <- (-1x30x1x1xf32, 120x30x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__4, parameter_133, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 120, 1, 1]

        # pd_op.reshape: (1x120x1x1xf32, 0x120xf32) <- (120xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_134, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x120x1x1xf32) <- (-1x120x1x1xf32, 1x120x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_23, reshape_18)

        # pd_op.hardsigmoid: (-1x120x1x1xf32) <- (-1x120x1x1xf32)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add__9, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x120x20x20xf32) <- (-1x120x20x20xf32, -1x120x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(concat_6, hardsigmoid_4)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x120x20x20xf32, 96x120x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(multiply__4, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__138)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_7 = split_2[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_11 = [slice_7, hardswish_16]

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_11, full_24)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_2 = paddle._C_ops.shape(concat_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_8, full_25, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_7, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_8, full_29, full_30, full_31]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(reshape__10, full_int_array_25, full_32)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_9 = split_3[1]

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x96x20x20xf32, 48x96x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(slice_9, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(hardswish_17, parameter_145, [1, 1], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_14 = [hardswish_17, batch_norm__150]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_14, full_33)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x20x20xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(concat_8, full_int_array_26, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(pool2d_6, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_26, reshape_20)

        # pd_op.relu_: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu__5 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__5, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_153, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_27, reshape_22)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add__11, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x20x20xf32) <- (-1x96x20x20xf32, -1x96x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(concat_8, hardsigmoid_5)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(multiply__5, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__156)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_10 = split_3[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_15 = [slice_10, hardswish_18]

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_15, full_34)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_3 = paddle._C_ops.shape(concat_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_3, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_11, full_35, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_9, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_11, full_39, full_40, full_41]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape__14, full_int_array_31, full_42)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_12 = split_4[1]

        # pd_op.conv2d: (-1x60x20x20xf32) <- (-1x96x20x20xf32, 60x96x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(slice_12, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x20x20xf32, 60xf32, 60xf32, xf32, xf32, None) <- (-1x60x20x20xf32, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x20x20xf32) <- (-1x60x20x20xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__162)

        # pd_op.depthwise_conv2d: (-1x60x20x20xf32) <- (-1x60x20x20xf32, 60x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(hardswish_19, parameter_164, [1, 1], [1, 1], 'EXPLICIT', 60, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x60x20x20xf32, 60xf32, 60xf32, xf32, xf32, None) <- (-1x60x20x20xf32, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x60x20x20xf32, -1x60x20x20xf32]) <- (-1x60x20x20xf32, -1x60x20x20xf32)
        combine_18 = [hardswish_19, batch_norm__168]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x20x20xf32) <- ([-1x60x20x20xf32, -1x60x20x20xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_18, full_43)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [1, 1]

        # pd_op.pool2d: (-1x120x1x1xf32) <- (-1x120x20x20xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(concat_10, full_int_array_32, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x30x1x1xf32) <- (-1x120x1x1xf32, 30x120x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(pool2d_7, parameter_169, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 30, 1, 1]

        # pd_op.reshape: (1x30x1x1xf32, 0x30xf32) <- (30xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_170, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x30x1x1xf32) <- (-1x30x1x1xf32, 1x30x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_30, reshape_24)

        # pd_op.relu_: (-1x30x1x1xf32) <- (-1x30x1x1xf32)
        relu__6 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x120x1x1xf32) <- (-1x30x1x1xf32, 120x30x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__6, parameter_171, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 120, 1, 1]

        # pd_op.reshape: (1x120x1x1xf32, 0x120xf32) <- (120xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_172, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x120x1x1xf32) <- (-1x120x1x1xf32, 1x120x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_31, reshape_26)

        # pd_op.hardsigmoid: (-1x120x1x1xf32) <- (-1x120x1x1xf32)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add__13, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x120x20x20xf32) <- (-1x120x20x20xf32, -1x120x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(concat_10, hardsigmoid_6)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x120x20x20xf32, 96x120x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(multiply__6, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__174)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_13 = split_4[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_19 = [slice_13, hardswish_20]

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_19, full_44)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_4 = paddle._C_ops.shape(concat_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_4, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_14, full_45, full_46, full_47, full_48]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_11, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_14, full_49, full_50, full_51]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_5 = paddle._C_ops.split(reshape__18, full_int_array_37, full_52)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_15 = split_5[1]

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x96x20x20xf32, 48x96x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(slice_15, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_179, parameter_180, parameter_181, parameter_182, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__180)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(hardswish_21, parameter_183, [1, 1], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_22 = [hardswish_21, batch_norm__186]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_22, full_53)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x20x20xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(concat_12, full_int_array_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(pool2d_8, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_34, reshape_28)

        # pd_op.relu_: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu__7 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__7, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_35, reshape_30)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add__15, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x20x20xf32) <- (-1x96x20x20xf32, -1x96x1x1xf32)
        multiply__7 = paddle._C_ops.multiply_(concat_12, hardsigmoid_7)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(multiply__7, parameter_192, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_193, parameter_194, parameter_195, parameter_196, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__192)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_16 = split_5[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_23 = [slice_16, hardswish_22]

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_23, full_54)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_5 = paddle._C_ops.shape(concat_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_5, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_17, full_55, full_56, full_57, full_58]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_13, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_17, full_59, full_60, full_61]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(reshape__22, full_int_array_43, full_62)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_18 = split_6[1]

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x96x20x20xf32, 48x96x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(slice_18, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_198, parameter_199, parameter_200, parameter_201, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__198)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(hardswish_23, parameter_202, [1, 1], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_203, parameter_204, parameter_205, parameter_206, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_26 = [hardswish_23, batch_norm__204]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_26, full_63)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x20x20xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(concat_14, full_int_array_44, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(pool2d_9, parameter_207, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_208, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_38, reshape_32)

        # pd_op.relu_: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu__8 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu__8, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_210, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_39, reshape_34)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_8 = paddle._C_ops.hardsigmoid(add__17, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x20x20xf32) <- (-1x96x20x20xf32, -1x96x1x1xf32)
        multiply__8 = paddle._C_ops.multiply_(concat_14, hardsigmoid_8)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(multiply__8, parameter_211, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__210)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_19 = split_6[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_27 = [slice_19, hardswish_24]

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_27, full_64)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_6 = paddle._C_ops.shape(concat_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_6, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_20, full_65, full_66, full_67, full_68]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_15, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_20, full_69, full_70, full_71]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_49 = [96, 96]

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x192x20x20xf32, 2xi64, 1xi32)
        split_7 = paddle._C_ops.split(reshape__26, full_int_array_49, full_72)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_21 = split_7[1]

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x96x20x20xf32, 48x96x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(slice_21, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_217, parameter_218, parameter_219, parameter_220, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__216)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(hardswish_25, parameter_221, [1, 1], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_13, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_30 = [hardswish_25, batch_norm__222]

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_30, full_73)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x20x20xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(concat_16, full_int_array_50, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(pool2d_10, parameter_226, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_227, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_42, reshape_36)

        # pd_op.relu_: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu__9 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__9, parameter_228, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_229, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_43, reshape_38)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_9 = paddle._C_ops.hardsigmoid(add__19, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x20x20xf32) <- (-1x96x20x20xf32, -1x96x1x1xf32)
        multiply__9 = paddle._C_ops.multiply_(concat_16, hardsigmoid_9)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(multiply__9, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__228)

        # builtin.slice: (-1x96x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32])
        slice_22 = split_7[0]

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_31 = [slice_22, hardswish_26]

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_31, full_74)

        # pd_op.shape: (4xi32) <- (-1x192x20x20xf32)
        shape_7 = paddle._C_ops.shape(concat_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_7, [0], full_int_array_53, full_int_array_54, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_23, full_75, full_76, full_77, full_78]

        # pd_op.reshape_: (-1x2x96x20x20xf32, 0x-1x192x20x20xf32) <- (-1x192x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_17, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x96x2x20x20xf32) <- (-1x2x96x20x20xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_23, full_79, full_80, full_81]

        # pd_op.reshape_: (-1x192x20x20xf32, 0x-1x96x2x20x20xf32) <- (-1x96x2x20x20xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x192x10x10xf32) <- (-1x192x20x20xf32, 192x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(reshape__30, parameter_235, [2, 2], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x192x10x10xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x10x10xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_14, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x192x10x10xf32) <- (-1x192x10x10xf32, 192x192x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(batch_norm__234, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x10x10xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x10x10xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x10x10xf32) <- (-1x192x10x10xf32)
        hardswish_27 = paddle._C_ops.hardswish(batch_norm__240)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x192x20x20xf32, 96x192x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(reshape__30, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_28 = paddle._C_ops.hardswish(batch_norm__246)

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x20x20xf32, 96x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(hardswish_28, parameter_250, [2, 2], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_15, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_55 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x10x10xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(batch_norm__252, full_int_array_55, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(pool2d_11, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_256, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_47, reshape_40)

        # pd_op.relu_: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu__10 = paddle._C_ops.relu_(add__20)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__10, parameter_257, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_258, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_48, reshape_42)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_10 = paddle._C_ops.hardsigmoid(add__21, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x10x10xf32) <- (-1x96x10x10xf32, -1x96x1x1xf32)
        multiply__10 = paddle._C_ops.multiply_(batch_norm__252, hardsigmoid_10)

        # pd_op.conv2d: (-1x192x10x10xf32) <- (-1x96x10x10xf32, 192x96x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(multiply__10, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x10x10xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x10x10xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x10x10xf32) <- (-1x192x10x10xf32)
        hardswish_29 = paddle._C_ops.hardswish(batch_norm__258)

        # builtin.combine: ([-1x192x10x10xf32, -1x192x10x10xf32]) <- (-1x192x10x10xf32, -1x192x10x10xf32)
        combine_34 = [hardswish_27, hardswish_29]

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_34, full_82)

        # pd_op.depthwise_conv2d: (-1x384x10x10xf32) <- (-1x384x10x10xf32, 384x1x3x3xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(concat_18, parameter_264, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x384x10x10xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x10x10xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_16, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x384x10x10xf32) <- (-1x384x10x10xf32)
        hardswish_30 = paddle._C_ops.hardswish(batch_norm__264)

        # pd_op.conv2d: (-1x384x10x10xf32) <- (-1x384x10x10xf32, 384x384x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(hardswish_30, parameter_269, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x10x10xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x10x10xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x384x10x10xf32) <- (-1x384x10x10xf32)
        hardswish_31 = paddle._C_ops.hardswish(batch_norm__270)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [192, 192]

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x192x10x10xf32, -1x192x10x10xf32]) <- (-1x384x10x10xf32, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(hardswish_31, full_int_array_58, full_83)

        # builtin.slice: (-1x192x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32])
        slice_24 = split_8[1]

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x192x10x10xf32, 96x192x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(slice_24, parameter_274, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_275, parameter_276, parameter_277, parameter_278, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_32 = paddle._C_ops.hardswish(batch_norm__276)

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x1x3x3xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(hardswish_32, parameter_279, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_17, parameter_280, parameter_281, parameter_282, parameter_283, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x96x10x10xf32, -1x96x10x10xf32]) <- (-1x96x10x10xf32, -1x96x10x10xf32)
        combine_35 = [hardswish_32, batch_norm__282]

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x10x10xf32) <- ([-1x96x10x10xf32, -1x96x10x10xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_35, full_84)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_59 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x10x10xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(concat_19, full_int_array_59, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(pool2d_12, parameter_284, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_285, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_52, reshape_44)

        # pd_op.relu_: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        relu__11 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__11, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_287, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_53, reshape_46)

        # pd_op.hardsigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        hardsigmoid_11 = paddle._C_ops.hardsigmoid(add__23, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x192x10x10xf32) <- (-1x192x10x10xf32, -1x192x1x1xf32)
        multiply__11 = paddle._C_ops.multiply_(concat_19, hardsigmoid_11)

        # pd_op.conv2d: (-1x192x10x10xf32) <- (-1x192x10x10xf32, 192x192x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(multiply__11, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x10x10xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x10x10xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_289, parameter_290, parameter_291, parameter_292, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x10x10xf32) <- (-1x192x10x10xf32)
        hardswish_33 = paddle._C_ops.hardswish(batch_norm__288)

        # builtin.slice: (-1x192x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32])
        slice_25 = split_8[0]

        # builtin.combine: ([-1x192x10x10xf32, -1x192x10x10xf32]) <- (-1x192x10x10xf32, -1x192x10x10xf32)
        combine_36 = [slice_25, hardswish_33]

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_36, full_85)

        # pd_op.shape: (4xi32) <- (-1x384x10x10xf32)
        shape_8 = paddle._C_ops.shape(concat_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_8, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_26, full_86, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x2x192x10x10xf32, 0x-1x384x10x10xf32) <- (-1x384x10x10xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_20, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x192x2x10x10xf32) <- (-1x2x192x10x10xf32)
        transpose_8 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_26, full_90, full_91, full_92]

        # pd_op.reshape_: (-1x384x10x10xf32, 0x-1x192x2x10x10xf32) <- (-1x192x2x10x10xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_64 = [192, 192]

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x192x10x10xf32, -1x192x10x10xf32]) <- (-1x384x10x10xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(reshape__34, full_int_array_64, full_93)

        # builtin.slice: (-1x192x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32])
        slice_27 = split_9[1]

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x192x10x10xf32, 96x192x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(slice_27, parameter_293, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_294, parameter_295, parameter_296, parameter_297, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_34 = paddle._C_ops.hardswish(batch_norm__294)

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x1x3x3xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(hardswish_34, parameter_298, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_18, parameter_299, parameter_300, parameter_301, parameter_302, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x96x10x10xf32, -1x96x10x10xf32]) <- (-1x96x10x10xf32, -1x96x10x10xf32)
        combine_39 = [hardswish_34, batch_norm__300]

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x10x10xf32) <- ([-1x96x10x10xf32, -1x96x10x10xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_39, full_94)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_65 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x10x10xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(concat_21, full_int_array_65, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(pool2d_13, parameter_303, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_56, reshape_48)

        # pd_op.relu_: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        relu__12 = paddle._C_ops.relu_(add__24)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu__12, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_306, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_57, reshape_50)

        # pd_op.hardsigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        hardsigmoid_12 = paddle._C_ops.hardsigmoid(add__25, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x192x10x10xf32) <- (-1x192x10x10xf32, -1x192x1x1xf32)
        multiply__12 = paddle._C_ops.multiply_(concat_21, hardsigmoid_12)

        # pd_op.conv2d: (-1x192x10x10xf32) <- (-1x192x10x10xf32, 192x192x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(multiply__12, parameter_307, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x10x10xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x10x10xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_308, parameter_309, parameter_310, parameter_311, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x192x10x10xf32) <- (-1x192x10x10xf32)
        hardswish_35 = paddle._C_ops.hardswish(batch_norm__306)

        # builtin.slice: (-1x192x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32])
        slice_28 = split_9[0]

        # builtin.combine: ([-1x192x10x10xf32, -1x192x10x10xf32]) <- (-1x192x10x10xf32, -1x192x10x10xf32)
        combine_40 = [slice_28, hardswish_35]

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x10x10xf32) <- ([-1x192x10x10xf32, -1x192x10x10xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_40, full_95)

        # pd_op.shape: (4xi32) <- (-1x384x10x10xf32)
        shape_9 = paddle._C_ops.shape(concat_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_9, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_29, full_96, full_97, full_98, full_99]

        # pd_op.reshape_: (-1x2x192x10x10xf32, 0x-1x384x10x10xf32) <- (-1x384x10x10xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_22, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x192x2x10x10xf32) <- (-1x2x192x10x10xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('10'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_29, full_100, full_101, full_102]

        # pd_op.reshape_: (-1x384x10x10xf32, 0x-1x192x2x10x10xf32) <- (-1x192x2x10x10xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(reshape__6, parameter_312, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_313, parameter_314, parameter_315, parameter_316, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_36 = paddle._C_ops.hardswish(batch_norm__312)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x192x20x20xf32, 96x192x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(reshape__30, parameter_317, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_318, parameter_319, parameter_320, parameter_321, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_37 = paddle._C_ops.hardswish(batch_norm__318)

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x384x10x10xf32, 96x384x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(reshape__38, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_323, parameter_324, parameter_325, parameter_326, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_38 = paddle._C_ops.hardswish(batch_norm__324)

        # pd_op.nearest_interp: (-1x96x20x20xf32) <- (-1x96x10x10xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(hardswish_38, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_43 = [nearest_interp_0, hardswish_37]

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_43, full_103)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x192x20x20xf32, 48x192x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(concat_23, parameter_327, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_328, parameter_329, parameter_330, parameter_331, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_39 = paddle._C_ops.hardswish(batch_norm__330)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x192x20x20xf32, 48x192x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(concat_23, parameter_332, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_333, parameter_334, parameter_335, parameter_336, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_40 = paddle._C_ops.hardswish(batch_norm__336)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x48x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(hardswish_40, parameter_337, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_338, parameter_339, parameter_340, parameter_341, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_41 = paddle._C_ops.hardswish(batch_norm__342)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x5x5xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(hardswish_41, parameter_342, [1, 1], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_19, parameter_343, parameter_344, parameter_345, parameter_346, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_42 = paddle._C_ops.hardswish(batch_norm__348)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x48x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(hardswish_42, parameter_347, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_348, parameter_349, parameter_350, parameter_351, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_43 = paddle._C_ops.hardswish(batch_norm__354)

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_44 = [hardswish_43, hardswish_39]

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_44, full_104)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(concat_24, parameter_352, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_353, parameter_354, parameter_355, parameter_356, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_44 = paddle._C_ops.hardswish(batch_norm__360)

        # pd_op.nearest_interp: (-1x96x40x40xf32) <- (-1x96x20x20xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(hardswish_44, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x96x40x40xf32, -1x96x40x40xf32]) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        combine_45 = [nearest_interp_1, hardswish_36]

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x40x40xf32) <- ([-1x96x40x40xf32, -1x96x40x40xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_45, full_105)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x192x40x40xf32, 48x192x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(concat_25, parameter_357, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_358, parameter_359, parameter_360, parameter_361, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_45 = paddle._C_ops.hardswish(batch_norm__366)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x192x40x40xf32, 48x192x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(concat_25, parameter_362, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_363, parameter_364, parameter_365, parameter_366, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_46 = paddle._C_ops.hardswish(batch_norm__372)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x48x40x40xf32, 48x48x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(hardswish_46, parameter_367, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_368, parameter_369, parameter_370, parameter_371, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_47 = paddle._C_ops.hardswish(batch_norm__378)

        # pd_op.depthwise_conv2d: (-1x48x40x40xf32) <- (-1x48x40x40xf32, 48x1x5x5xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(hardswish_47, parameter_372, [1, 1], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_20, parameter_373, parameter_374, parameter_375, parameter_376, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_48 = paddle._C_ops.hardswish(batch_norm__384)

        # pd_op.conv2d: (-1x48x40x40xf32) <- (-1x48x40x40xf32, 48x48x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(hardswish_48, parameter_377, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x40x40xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x40x40xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_378, parameter_379, parameter_380, parameter_381, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x40x40xf32) <- (-1x48x40x40xf32)
        hardswish_49 = paddle._C_ops.hardswish(batch_norm__390)

        # builtin.combine: ([-1x48x40x40xf32, -1x48x40x40xf32]) <- (-1x48x40x40xf32, -1x48x40x40xf32)
        combine_46 = [hardswish_49, hardswish_45]

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x40x40xf32) <- ([-1x48x40x40xf32, -1x48x40x40xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_46, full_106)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(concat_26, parameter_382, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_383, parameter_384, parameter_385, parameter_386, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_50 = paddle._C_ops.hardswish(batch_norm__396)

        # pd_op.depthwise_conv2d: (-1x96x20x20xf32) <- (-1x96x40x40xf32, 96x1x5x5xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(hardswish_50, parameter_387, [2, 2], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_21, parameter_388, parameter_389, parameter_390, parameter_391, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_51 = paddle._C_ops.hardswish(batch_norm__402)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(hardswish_51, parameter_392, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_393, parameter_394, parameter_395, parameter_396, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_52 = paddle._C_ops.hardswish(batch_norm__408)

        # builtin.combine: ([-1x96x20x20xf32, -1x96x20x20xf32]) <- (-1x96x20x20xf32, -1x96x20x20xf32)
        combine_47 = [hardswish_52, hardswish_44]

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x20x20xf32) <- ([-1x96x20x20xf32, -1x96x20x20xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_47, full_107)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x192x20x20xf32, 48x192x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(concat_27, parameter_397, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_398, parameter_399, parameter_400, parameter_401, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_53 = paddle._C_ops.hardswish(batch_norm__414)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x192x20x20xf32, 48x192x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(concat_27, parameter_402, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_403, parameter_404, parameter_405, parameter_406, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_54 = paddle._C_ops.hardswish(batch_norm__420)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x48x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(hardswish_54, parameter_407, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_408, parameter_409, parameter_410, parameter_411, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_55 = paddle._C_ops.hardswish(batch_norm__426)

        # pd_op.depthwise_conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x1x5x5xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(hardswish_55, parameter_412, [1, 1], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_22, parameter_413, parameter_414, parameter_415, parameter_416, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_56 = paddle._C_ops.hardswish(batch_norm__432)

        # pd_op.conv2d: (-1x48x20x20xf32) <- (-1x48x20x20xf32, 48x48x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(hardswish_56, parameter_417, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x20x20xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x20x20xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_418, parameter_419, parameter_420, parameter_421, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x20x20xf32) <- (-1x48x20x20xf32)
        hardswish_57 = paddle._C_ops.hardswish(batch_norm__438)

        # builtin.combine: ([-1x48x20x20xf32, -1x48x20x20xf32]) <- (-1x48x20x20xf32, -1x48x20x20xf32)
        combine_48 = [hardswish_57, hardswish_53]

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x20x20xf32) <- ([-1x48x20x20xf32, -1x48x20x20xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_48, full_108)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(concat_28, parameter_422, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_423, parameter_424, parameter_425, parameter_426, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_58 = paddle._C_ops.hardswish(batch_norm__444)

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x20x20xf32, 96x1x5x5xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(hardswish_58, parameter_427, [2, 2], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_23, parameter_428, parameter_429, parameter_430, parameter_431, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_59 = paddle._C_ops.hardswish(batch_norm__450)

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x96x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(hardswish_59, parameter_432, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_433, parameter_434, parameter_435, parameter_436, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_60 = paddle._C_ops.hardswish(batch_norm__456)

        # builtin.combine: ([-1x96x10x10xf32, -1x96x10x10xf32]) <- (-1x96x10x10xf32, -1x96x10x10xf32)
        combine_49 = [hardswish_60, hardswish_38]

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x10x10xf32) <- ([-1x96x10x10xf32, -1x96x10x10xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_49, full_109)

        # pd_op.conv2d: (-1x48x10x10xf32) <- (-1x192x10x10xf32, 48x192x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(concat_29, parameter_437, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x10x10xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x10x10xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_438, parameter_439, parameter_440, parameter_441, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x10x10xf32) <- (-1x48x10x10xf32)
        hardswish_61 = paddle._C_ops.hardswish(batch_norm__462)

        # pd_op.conv2d: (-1x48x10x10xf32) <- (-1x192x10x10xf32, 48x192x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(concat_29, parameter_442, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x10x10xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x10x10xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_443, parameter_444, parameter_445, parameter_446, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x10x10xf32) <- (-1x48x10x10xf32)
        hardswish_62 = paddle._C_ops.hardswish(batch_norm__468)

        # pd_op.conv2d: (-1x48x10x10xf32) <- (-1x48x10x10xf32, 48x48x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(hardswish_62, parameter_447, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x10x10xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x10x10xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_448, parameter_449, parameter_450, parameter_451, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x10x10xf32) <- (-1x48x10x10xf32)
        hardswish_63 = paddle._C_ops.hardswish(batch_norm__474)

        # pd_op.depthwise_conv2d: (-1x48x10x10xf32) <- (-1x48x10x10xf32, 48x1x5x5xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(hardswish_63, parameter_452, [1, 1], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x48x10x10xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x10x10xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_24, parameter_453, parameter_454, parameter_455, parameter_456, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x10x10xf32) <- (-1x48x10x10xf32)
        hardswish_64 = paddle._C_ops.hardswish(batch_norm__480)

        # pd_op.conv2d: (-1x48x10x10xf32) <- (-1x48x10x10xf32, 48x48x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(hardswish_64, parameter_457, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x10x10xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x10x10xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_458, parameter_459, parameter_460, parameter_461, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x48x10x10xf32) <- (-1x48x10x10xf32)
        hardswish_65 = paddle._C_ops.hardswish(batch_norm__486)

        # builtin.combine: ([-1x48x10x10xf32, -1x48x10x10xf32]) <- (-1x48x10x10xf32, -1x48x10x10xf32)
        combine_50 = [hardswish_65, hardswish_61]

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x10x10xf32) <- ([-1x48x10x10xf32, -1x48x10x10xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_50, full_110)

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x96x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(concat_30, parameter_462, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_463, parameter_464, parameter_465, parameter_466, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_66 = paddle._C_ops.hardswish(batch_norm__492)

        # pd_op.depthwise_conv2d: (-1x96x5x5xf32) <- (-1x96x10x10xf32, 96x1x5x5xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(hardswish_38, parameter_467, [2, 2], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_25, parameter_468, parameter_469, parameter_470, parameter_471, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_67 = paddle._C_ops.hardswish(batch_norm__498)

        # pd_op.conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x96x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(hardswish_67, parameter_472, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_473, parameter_474, parameter_475, parameter_476, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_68 = paddle._C_ops.hardswish(batch_norm__504)

        # pd_op.depthwise_conv2d: (-1x96x5x5xf32) <- (-1x96x10x10xf32, 96x1x5x5xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(hardswish_66, parameter_477, [2, 2], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_26, parameter_478, parameter_479, parameter_480, parameter_481, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_69 = paddle._C_ops.hardswish(batch_norm__510)

        # pd_op.conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x96x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(hardswish_69, parameter_482, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_483, parameter_484, parameter_485, parameter_486, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_70 = paddle._C_ops.hardswish(batch_norm__516)

        # pd_op.add_: (-1x96x5x5xf32) <- (-1x96x5x5xf32, -1x96x5x5xf32)
        add__26 = paddle._C_ops.add_(hardswish_68, hardswish_70)

        # pd_op.depthwise_conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x1x5x5xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(hardswish_50, parameter_487, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_27, parameter_488, parameter_489, parameter_490, parameter_491, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_71 = paddle._C_ops.hardswish(batch_norm__522)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(hardswish_71, parameter_492, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_493, parameter_494, parameter_495, parameter_496, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_72 = paddle._C_ops.hardswish(batch_norm__528)

        # pd_op.depthwise_conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x1x5x5xf32)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(hardswish_72, parameter_497, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_28, parameter_498, parameter_499, parameter_500, parameter_501, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_73 = paddle._C_ops.hardswish(batch_norm__534)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(hardswish_73, parameter_502, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_503, parameter_504, parameter_505, parameter_506, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        hardswish_74 = paddle._C_ops.hardswish(batch_norm__540)

        # pd_op.conv2d: (-1x112x40x40xf32) <- (-1x96x40x40xf32, 112x96x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(hardswish_74, parameter_507, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_70 = [1, 112, 1, 1]

        # pd_op.reshape: (1x112x1x1xf32, 0x112xf32) <- (112xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_508, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x112x40x40xf32) <- (-1x112x40x40xf32, 1x112x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_88, reshape_52)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_71 = [80, 32]

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x80x40x40xf32, -1x32x40x40xf32]) <- (-1x112x40x40xf32, 2xi64, 1xi32)
        split_10 = paddle._C_ops.split(add__27, full_int_array_71, full_111)

        # builtin.slice: (-1x80x40x40xf32) <- ([-1x80x40x40xf32, -1x32x40x40xf32])
        slice_30 = split_10[0]

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_72 = [-1, 80, 1600]

        # pd_op.reshape_: (-1x80x1600xf32, 0x-1x80x40x40xf32) <- (-1x80x40x40xf32, 3xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_30, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x1600xf32) <- (-1x80x1600xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(reshape__40)

        # builtin.slice: (-1x32x40x40xf32) <- ([-1x80x40x40xf32, -1x32x40x40xf32])
        slice_31 = split_10[1]

        # pd_op.transpose: (-1x40x40x32xf32) <- (-1x32x40x40xf32)
        transpose_10 = paddle._C_ops.transpose(slice_31, [0, 2, 3, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_73 = [-1, 8]

        # pd_op.reshape_: (-1x8xf32, 0x-1x40x40x32xf32) <- (-1x40x40x32xf32, 2xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, full_int_array_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x8xf32) <- (-1x8xf32)
        softmax__0 = paddle._C_ops.softmax_(reshape__42, 1)

        # pd_op.matmul: (-1xf32) <- (-1x8xf32, 8xf32)
        matmul_0 = paddle._C_ops.matmul(softmax__0, parameter_509, False, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_74 = [-1, 1600, 4]

        # pd_op.reshape_: (-1x1600x4xf32, 0x-1xf32) <- (-1xf32, 3xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_0, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x1x5x5xf32)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(hardswish_58, parameter_510, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_29, parameter_511, parameter_512, parameter_513, parameter_514, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_75 = paddle._C_ops.hardswish(batch_norm__546)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_89 = paddle._C_ops.conv2d(hardswish_75, parameter_515, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_516, parameter_517, parameter_518, parameter_519, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_76 = paddle._C_ops.hardswish(batch_norm__552)

        # pd_op.depthwise_conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x1x5x5xf32)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(hardswish_76, parameter_520, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_30, parameter_521, parameter_522, parameter_523, parameter_524, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_77 = paddle._C_ops.hardswish(batch_norm__558)

        # pd_op.conv2d: (-1x96x20x20xf32) <- (-1x96x20x20xf32, 96x96x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(hardswish_77, parameter_525, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x20x20xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x20x20xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_526, parameter_527, parameter_528, parameter_529, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x20x20xf32) <- (-1x96x20x20xf32)
        hardswish_78 = paddle._C_ops.hardswish(batch_norm__564)

        # pd_op.conv2d: (-1x112x20x20xf32) <- (-1x96x20x20xf32, 112x96x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(hardswish_78, parameter_530, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_75 = [1, 112, 1, 1]

        # pd_op.reshape: (1x112x1x1xf32, 0x112xf32) <- (112xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_531, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x112x20x20xf32) <- (-1x112x20x20xf32, 1x112x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_91, reshape_54)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_76 = [80, 32]

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x80x20x20xf32, -1x32x20x20xf32]) <- (-1x112x20x20xf32, 2xi64, 1xi32)
        split_11 = paddle._C_ops.split(add__28, full_int_array_76, full_112)

        # builtin.slice: (-1x80x20x20xf32) <- ([-1x80x20x20xf32, -1x32x20x20xf32])
        slice_32 = split_11[0]

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_77 = [-1, 80, 400]

        # pd_op.reshape_: (-1x80x400xf32, 0x-1x80x20x20xf32) <- (-1x80x20x20xf32, 3xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_32, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x400xf32) <- (-1x80x400xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(reshape__46)

        # builtin.slice: (-1x32x20x20xf32) <- ([-1x80x20x20xf32, -1x32x20x20xf32])
        slice_33 = split_11[1]

        # pd_op.transpose: (-1x20x20x32xf32) <- (-1x32x20x20xf32)
        transpose_11 = paddle._C_ops.transpose(slice_33, [0, 2, 3, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_78 = [-1, 8]

        # pd_op.reshape_: (-1x8xf32, 0x-1x20x20x32xf32) <- (-1x20x20x32xf32, 2xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x8xf32) <- (-1x8xf32)
        softmax__1 = paddle._C_ops.softmax_(reshape__48, 1)

        # pd_op.matmul: (-1xf32) <- (-1x8xf32, 8xf32)
        matmul_1 = paddle._C_ops.matmul(softmax__1, parameter_509, False, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_79 = [-1, 400, 4]

        # pd_op.reshape_: (-1x400x4xf32, 0x-1xf32) <- (-1xf32, 3xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_1, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x1x5x5xf32)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(hardswish_66, parameter_532, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_31, parameter_533, parameter_534, parameter_535, parameter_536, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_79 = paddle._C_ops.hardswish(batch_norm__570)

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x96x1x1xf32)
        conv2d_92 = paddle._C_ops.conv2d(hardswish_79, parameter_537, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_538, parameter_539, parameter_540, parameter_541, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_80 = paddle._C_ops.hardswish(batch_norm__576)

        # pd_op.depthwise_conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x1x5x5xf32)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(hardswish_80, parameter_542, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_32, parameter_543, parameter_544, parameter_545, parameter_546, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_81 = paddle._C_ops.hardswish(batch_norm__582)

        # pd_op.conv2d: (-1x96x10x10xf32) <- (-1x96x10x10xf32, 96x96x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(hardswish_81, parameter_547, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x10x10xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x10x10xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_548, parameter_549, parameter_550, parameter_551, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x10x10xf32) <- (-1x96x10x10xf32)
        hardswish_82 = paddle._C_ops.hardswish(batch_norm__588)

        # pd_op.conv2d: (-1x112x10x10xf32) <- (-1x96x10x10xf32, 112x96x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(hardswish_82, parameter_552, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [1, 112, 1, 1]

        # pd_op.reshape: (1x112x1x1xf32, 0x112xf32) <- (112xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_553, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x112x10x10xf32) <- (-1x112x10x10xf32, 1x112x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_94, reshape_56)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_81 = [80, 32]

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x80x10x10xf32, -1x32x10x10xf32]) <- (-1x112x10x10xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add__29, full_int_array_81, full_113)

        # builtin.slice: (-1x80x10x10xf32) <- ([-1x80x10x10xf32, -1x32x10x10xf32])
        slice_34 = split_12[0]

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_82 = [-1, 80, 100]

        # pd_op.reshape_: (-1x80x100xf32, 0x-1x80x10x10xf32) <- (-1x80x10x10xf32, 3xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_34, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x100xf32) <- (-1x80x100xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(reshape__52)

        # builtin.slice: (-1x32x10x10xf32) <- ([-1x80x10x10xf32, -1x32x10x10xf32])
        slice_35 = split_12[1]

        # pd_op.transpose: (-1x10x10x32xf32) <- (-1x32x10x10xf32)
        transpose_12 = paddle._C_ops.transpose(slice_35, [0, 2, 3, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_83 = [-1, 8]

        # pd_op.reshape_: (-1x8xf32, 0x-1x10x10x32xf32) <- (-1x10x10x32xf32, 2xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x8xf32) <- (-1x8xf32)
        softmax__2 = paddle._C_ops.softmax_(reshape__54, 1)

        # pd_op.matmul: (-1xf32) <- (-1x8xf32, 8xf32)
        matmul_2 = paddle._C_ops.matmul(softmax__2, parameter_509, False, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_84 = [-1, 100, 4]

        # pd_op.reshape_: (-1x100x4xf32, 0x-1xf32) <- (-1xf32, 3xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_2, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x1x5x5xf32)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(add__26, parameter_554, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_33, parameter_555, parameter_556, parameter_557, parameter_558, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_83 = paddle._C_ops.hardswish(batch_norm__594)

        # pd_op.conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x96x1x1xf32)
        conv2d_95 = paddle._C_ops.conv2d(hardswish_83, parameter_559, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_560, parameter_561, parameter_562, parameter_563, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_84 = paddle._C_ops.hardswish(batch_norm__600)

        # pd_op.depthwise_conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x1x5x5xf32)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(hardswish_84, parameter_564, [1, 1], [2, 2], 'EXPLICIT', 96, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_34, parameter_565, parameter_566, parameter_567, parameter_568, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_85 = paddle._C_ops.hardswish(batch_norm__606)

        # pd_op.conv2d: (-1x96x5x5xf32) <- (-1x96x5x5xf32, 96x96x1x1xf32)
        conv2d_96 = paddle._C_ops.conv2d(hardswish_85, parameter_569, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x5x5xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_570, parameter_571, parameter_572, parameter_573, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x96x5x5xf32) <- (-1x96x5x5xf32)
        hardswish_86 = paddle._C_ops.hardswish(batch_norm__612)

        # pd_op.conv2d: (-1x112x5x5xf32) <- (-1x96x5x5xf32, 112x96x1x1xf32)
        conv2d_97 = paddle._C_ops.conv2d(hardswish_86, parameter_574, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 112, 1, 1]

        # pd_op.reshape: (1x112x1x1xf32, 0x112xf32) <- (112xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_575, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x112x5x5xf32) <- (-1x112x5x5xf32, 1x112x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_97, reshape_58)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_86 = [80, 32]

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x80x5x5xf32, -1x32x5x5xf32]) <- (-1x112x5x5xf32, 2xi64, 1xi32)
        split_13 = paddle._C_ops.split(add__30, full_int_array_86, full_114)

        # builtin.slice: (-1x80x5x5xf32) <- ([-1x80x5x5xf32, -1x32x5x5xf32])
        slice_36 = split_13[0]

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_87 = [-1, 80, 25]

        # pd_op.reshape_: (-1x80x25xf32, 0x-1x80x5x5xf32) <- (-1x80x5x5xf32, 3xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_36, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x25xf32) <- (-1x80x25xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(reshape__58)

        # builtin.slice: (-1x32x5x5xf32) <- ([-1x80x5x5xf32, -1x32x5x5xf32])
        slice_37 = split_13[1]

        # pd_op.transpose: (-1x5x5x32xf32) <- (-1x32x5x5xf32)
        transpose_13 = paddle._C_ops.transpose(slice_37, [0, 2, 3, 1])

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_88 = [-1, 8]

        # pd_op.reshape_: (-1x8xf32, 0x-1x5x5x32xf32) <- (-1x5x5x32xf32, 2xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x8xf32) <- (-1x8xf32)
        softmax__3 = paddle._C_ops.softmax_(reshape__60, 1)

        # pd_op.matmul: (-1xf32) <- (-1x8xf32, 8xf32)
        matmul_3 = paddle._C_ops.matmul(softmax__3, parameter_509, False, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_89 = [-1, 25, 4]

        # pd_op.reshape_: (-1x25x4xf32, 0x-1xf32) <- (-1xf32, 3xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_3, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x80x1600xf32, -1x80x400xf32, -1x80x100xf32, -1x80x25xf32]) <- (-1x80x1600xf32, -1x80x400xf32, -1x80x100xf32, -1x80x25xf32)
        combine_51 = [sigmoid__0, sigmoid__1, sigmoid__2, sigmoid__3]

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x2125xf32) <- ([-1x80x1600xf32, -1x80x400xf32, -1x80x100xf32, -1x80x25xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_51, full_115)

        # builtin.combine: ([-1x1600x4xf32, -1x400x4xf32, -1x100x4xf32, -1x25x4xf32]) <- (-1x1600x4xf32, -1x400x4xf32, -1x100x4xf32, -1x25x4xf32)
        combine_52 = [reshape__44, reshape__50, reshape__56, reshape__62]

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2125x4xf32) <- ([-1x1600x4xf32, -1x400x4xf32, -1x100x4xf32, -1x25x4xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_52, full_116)

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x2125x2xf32, -1x2125x2xf32]) <- (-1x2125x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_32, 2, full_117)

        # builtin.slice: (-1x2125x2xf32) <- ([-1x2125x2xf32, -1x2125x2xf32])
        slice_38 = split_with_num_0[0]

        # pd_op.full: (1xf32) <- ()
        full_118 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2125x2xf32) <- (-1x2125x2xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_38, full_118, float('0'), True)

        # pd_op.add_: (-1x2125x2xf32) <- (-1x2125x2xf32, 2125x2xf32)
        add__31 = paddle._C_ops.add_(scale__0, parameter_576)

        # builtin.slice: (-1x2125x2xf32) <- ([-1x2125x2xf32, -1x2125x2xf32])
        slice_39 = split_with_num_0[1]

        # pd_op.add_: (-1x2125x2xf32) <- (-1x2125x2xf32, 2125x2xf32)
        add__32 = paddle._C_ops.add_(slice_39, parameter_576)

        # builtin.combine: ([-1x2125x2xf32, -1x2125x2xf32]) <- (-1x2125x2xf32, -1x2125x2xf32)
        combine_53 = [add__31, add__32]

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2125x4xf32) <- ([-1x2125x2xf32, -1x2125x2xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_53, full_119)

        # pd_op.multiply_: (-1x2125x4xf32) <- (-1x2125x4xf32, 2125x1xf32)
        multiply__13 = paddle._C_ops.multiply_(concat_33, parameter_577)

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32]) <- (1x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(feed_1, 2, full_120)

        # builtin.slice: (1x1xf32) <- ([1x1xf32, 1x1xf32])
        slice_40 = split_with_num_1[1]

        # builtin.slice: (1x1xf32) <- ([1x1xf32, 1x1xf32])
        slice_41 = split_with_num_1[0]

        # builtin.combine: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32)
        combine_54 = [slice_40, slice_41, slice_40, slice_41]

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x4xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_54, full_121)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_90 = [-1, 1, 4]

        # pd_op.reshape_: (1x1x4xf32, 0x1x4xf32) <- (1x4xf32, 3xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_34, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.divide_: (-1x2125x4xf32) <- (-1x2125x4xf32, 1x1x4xf32)
        divide__0 = paddle._C_ops.divide_(multiply__13, reshape__64)

        # pd_op.multiclass_nms3: (-1x6xf32, -1x1xi32, -1xi32) <- (-1x2125x4xf32, -1x80x2125xf32, None)
        multiclass_nms3_0, multiclass_nms3_1, multiclass_nms3_2 = (lambda x, f: f(x))(paddle._C_ops.multiclass_nms3(divide__0, concat_31, None, float('0.025'), 1000, 100, float('0.6'), True, float('1'), -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))
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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_356, parameter_353, parameter_355, parameter_354, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_406, parameter_403, parameter_405, parameter_404, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_416, parameter_413, parameter_415, parameter_414, parameter_417, parameter_421, parameter_418, parameter_420, parameter_419, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_431, parameter_428, parameter_430, parameter_429, parameter_432, parameter_436, parameter_433, parameter_435, parameter_434, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_456, parameter_453, parameter_455, parameter_454, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_531, parameter_532, parameter_536, parameter_533, parameter_535, parameter_534, parameter_537, parameter_541, parameter_538, parameter_540, parameter_539, parameter_542, parameter_546, parameter_543, parameter_545, parameter_544, parameter_547, parameter_551, parameter_548, parameter_550, parameter_549, parameter_552, parameter_553, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_575, parameter_576, parameter_577, feed_1, feed_0):
        return self.builtin_module_1463_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_321, parameter_318, parameter_320, parameter_319, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_341, parameter_338, parameter_340, parameter_339, parameter_342, parameter_346, parameter_343, parameter_345, parameter_344, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_356, parameter_353, parameter_355, parameter_354, parameter_357, parameter_361, parameter_358, parameter_360, parameter_359, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_406, parameter_403, parameter_405, parameter_404, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_416, parameter_413, parameter_415, parameter_414, parameter_417, parameter_421, parameter_418, parameter_420, parameter_419, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_431, parameter_428, parameter_430, parameter_429, parameter_432, parameter_436, parameter_433, parameter_435, parameter_434, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_456, parameter_453, parameter_455, parameter_454, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_531, parameter_532, parameter_536, parameter_533, parameter_535, parameter_534, parameter_537, parameter_541, parameter_538, parameter_540, parameter_539, parameter_542, parameter_546, parameter_543, parameter_545, parameter_544, parameter_547, parameter_551, parameter_548, parameter_550, parameter_549, parameter_552, parameter_553, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_575, parameter_576, parameter_577, feed_1, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1463_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([24, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([48, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([44, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([44, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([11], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([44], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([48, 44, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([24, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([24, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([12], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([96, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([60, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([60, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([96, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([60, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([60, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([30], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([96, 120, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([192, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([96, 1, 5, 5], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([2125, 2], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([2125, 1], dtype='float32', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1.0, 1.0], dtype='float32').reshape([1, 2]),
            # feed_0
            paddle.uniform([1, 3, 320, 320], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[24, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[48, 24, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[44, 24, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[44, 1, 3, 3], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[11, 44, 1, 1], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[11], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[44, 11, 1, 1], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[44], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[48, 44, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[24, 48, 1, 1], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[48, 12, 1, 1], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[24, 48, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[48, 12, 1, 1], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[12, 48, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[12], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[48, 12, 1, 1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[96, 48, 1, 1], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[60, 96, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[60, 1, 3, 3], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[30, 120, 1, 1], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[120, 30, 1, 1], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[96, 120, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[60, 96, 1, 1], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[60, 1, 3, 3], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[30, 120, 1, 1], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[30], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[120, 30, 1, 1], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[96, 120, 1, 1], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[192, 48, 1, 1], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[192, 48, 1, 1], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[96, 384, 1, 1], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[48, 192, 1, 1], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[112, 96, 1, 1], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[112, 96, 1, 1], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[112, 96, 1, 1], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[96, 1, 5, 5], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[112, 96, 1, 1], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[2125, 2], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[2125, 1], dtype='float32'),
            # feed_1
            paddle.static.InputSpec(shape=[1, 2], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 320, 320], dtype='float32'),
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