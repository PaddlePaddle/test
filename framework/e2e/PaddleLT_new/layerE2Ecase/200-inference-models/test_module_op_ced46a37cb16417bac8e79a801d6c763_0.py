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
    return [779][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1440_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_341, parameter_342, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_417, parameter_418, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_436, parameter_437, parameter_438, parameter_442, parameter_439, parameter_441, parameter_440, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_454, parameter_455, parameter_456, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_473, parameter_474, parameter_475, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_566, parameter_563, parameter_565, parameter_564, parameter_567, parameter_568, parameter_569, parameter_570, parameter_571, parameter_575, parameter_572, parameter_574, parameter_573, parameter_576, parameter_580, parameter_577, parameter_579, parameter_578, parameter_581, parameter_585, parameter_582, parameter_584, parameter_583, parameter_586, parameter_587, parameter_588, parameter_589, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_606, parameter_607, parameter_608, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_630, parameter_631, parameter_632, parameter_633, parameter_637, parameter_634, parameter_636, parameter_635, parameter_638, parameter_642, parameter_639, parameter_641, parameter_640, parameter_643, parameter_647, parameter_644, parameter_646, parameter_645, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_653, feed_0):

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x3x224x224xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x64x56x56xf32, 128x64x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 128x4x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x128x56x56xf32, 256x128x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x56x56xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(batch_norm__18, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 3]

        # pd_op.squeeze_: (-1x256xf32, None) <- (-1x256x1x1xf32, 2xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x256xf32, 256x16xf32)
        matmul_0 = paddle._C_ops.matmul(squeeze__0, parameter_20, False, False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__0 = paddle._C_ops.add_(matmul_0, parameter_21)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__3 = paddle._C_ops.relu_(add__0)

        # pd_op.matmul: (-1x256xf32) <- (-1x16xf32, 16x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu__3, parameter_22, False, False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__1 = paddle._C_ops.add_(matmul_1, parameter_23)

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 3]

        # pd_op.unsqueeze_: (-1x256x1x1xf32, None) <- (-1x256xf32, 2xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__0, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(batch_norm__18, unsqueeze__0)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_0, parameter_24, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_25, parameter_26, parameter_27, parameter_28, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__2 = paddle._C_ops.add_(batch_norm__24, multiply__0)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__4 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__4, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 128x4x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_35, parameter_36, parameter_37, parameter_38, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x128x56x56xf32, 256x128x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__6, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x56x56xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(batch_norm__42, full_int_array_4, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [2, 3]

        # pd_op.squeeze_: (-1x256xf32, None) <- (-1x256x1x1xf32, 2xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_2, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x256xf32, 256x16xf32)
        matmul_2 = paddle._C_ops.matmul(squeeze__2, parameter_44, False, False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__3 = paddle._C_ops.add_(matmul_2, parameter_45)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__7 = paddle._C_ops.relu_(add__3)

        # pd_op.matmul: (-1x256xf32) <- (-1x16xf32, 16x256xf32)
        matmul_3 = paddle._C_ops.matmul(relu__7, parameter_46, False, False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__4 = paddle._C_ops.add_(matmul_3, parameter_47)

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [2, 3]

        # pd_op.unsqueeze_: (-1x256x1x1xf32, None) <- (-1x256xf32, 2xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__1, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(batch_norm__42, unsqueeze__2)

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__5 = paddle._C_ops.add_(relu__4, multiply__1)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__8 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__8, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_49, parameter_50, parameter_51, parameter_52, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 128x4x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__9, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_54, parameter_55, parameter_56, parameter_57, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__10 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x128x56x56xf32, 256x128x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__10, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_59, parameter_60, parameter_61, parameter_62, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x56x56xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(batch_norm__60, full_int_array_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [2, 3]

        # pd_op.squeeze_: (-1x256xf32, None) <- (-1x256x1x1xf32, 2xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_3, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16xf32) <- (-1x256xf32, 256x16xf32)
        matmul_4 = paddle._C_ops.matmul(squeeze__4, parameter_63, False, False)

        # pd_op.add_: (-1x16xf32) <- (-1x16xf32, 16xf32)
        add__6 = paddle._C_ops.add_(matmul_4, parameter_64)

        # pd_op.relu_: (-1x16xf32) <- (-1x16xf32)
        relu__11 = paddle._C_ops.relu_(add__6)

        # pd_op.matmul: (-1x256xf32) <- (-1x16xf32, 16x256xf32)
        matmul_5 = paddle._C_ops.matmul(relu__11, parameter_65, False, False)

        # pd_op.add_: (-1x256xf32) <- (-1x256xf32, 256xf32)
        add__7 = paddle._C_ops.add_(matmul_5, parameter_66)

        # pd_op.sigmoid_: (-1x256xf32) <- (-1x256xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__7)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [2, 3]

        # pd_op.unsqueeze_: (-1x256x1x1xf32, None) <- (-1x256xf32, 2xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__2, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(batch_norm__60, unsqueeze__4)

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__8 = paddle._C_ops.add_(relu__8, multiply__2)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__12 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x256x56x56xf32, 256x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__12, parameter_67, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__13 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x256x56x56xf32, 256x8x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__13, parameter_72, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu__14, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x28x28xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(batch_norm__78, full_int_array_10, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [2, 3]

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x512x1x1xf32, 2xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_4, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x32xf32) <- (-1x512xf32, 512x32xf32)
        matmul_6 = paddle._C_ops.matmul(squeeze__6, parameter_82, False, False)

        # pd_op.add_: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add__9 = paddle._C_ops.add_(matmul_6, parameter_83)

        # pd_op.relu_: (-1x32xf32) <- (-1x32xf32)
        relu__15 = paddle._C_ops.relu_(add__9)

        # pd_op.matmul: (-1x512xf32) <- (-1x32xf32, 32x512xf32)
        matmul_7 = paddle._C_ops.matmul(relu__15, parameter_84, False, False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__10 = paddle._C_ops.add_(matmul_7, parameter_85)

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(add__10)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [2, 3]

        # pd_op.unsqueeze_: (-1x512x1x1xf32, None) <- (-1x512xf32, 2xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__3, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(batch_norm__78, unsqueeze__6)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x56x56xf32, 512x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__12, parameter_86, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__11 = paddle._C_ops.add_(batch_norm__84, multiply__3)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__16 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__16, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__17 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 256x8x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__17, parameter_96, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__18, parameter_101, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_102, parameter_103, parameter_104, parameter_105, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x28x28xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(batch_norm__102, full_int_array_13, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [2, 3]

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x512x1x1xf32, 2xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_5, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x32xf32) <- (-1x512xf32, 512x32xf32)
        matmul_8 = paddle._C_ops.matmul(squeeze__8, parameter_106, False, False)

        # pd_op.add_: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add__12 = paddle._C_ops.add_(matmul_8, parameter_107)

        # pd_op.relu_: (-1x32xf32) <- (-1x32xf32)
        relu__19 = paddle._C_ops.relu_(add__12)

        # pd_op.matmul: (-1x512xf32) <- (-1x32xf32, 32x512xf32)
        matmul_9 = paddle._C_ops.matmul(relu__19, parameter_108, False, False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__13 = paddle._C_ops.add_(matmul_9, parameter_109)

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(add__13)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [2, 3]

        # pd_op.unsqueeze_: (-1x512x1x1xf32, None) <- (-1x512xf32, 2xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__4, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(batch_norm__102, unsqueeze__8)

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__14 = paddle._C_ops.add_(relu__16, multiply__4)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__20 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__20, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__21 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 256x8x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__21, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__22 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__22, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x28x28xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(batch_norm__120, full_int_array_16, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [2, 3]

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x512x1x1xf32, 2xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_6, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x32xf32) <- (-1x512xf32, 512x32xf32)
        matmul_10 = paddle._C_ops.matmul(squeeze__10, parameter_125, False, False)

        # pd_op.add_: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add__15 = paddle._C_ops.add_(matmul_10, parameter_126)

        # pd_op.relu_: (-1x32xf32) <- (-1x32xf32)
        relu__23 = paddle._C_ops.relu_(add__15)

        # pd_op.matmul: (-1x512xf32) <- (-1x32xf32, 32x512xf32)
        matmul_11 = paddle._C_ops.matmul(relu__23, parameter_127, False, False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__16 = paddle._C_ops.add_(matmul_11, parameter_128)

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(add__16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [2, 3]

        # pd_op.unsqueeze_: (-1x512x1x1xf32, None) <- (-1x512xf32, 2xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__5, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(batch_norm__120, unsqueeze__10)

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__17 = paddle._C_ops.add_(relu__20, multiply__5)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__24 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__24, parameter_129, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__25 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 256x8x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__25, parameter_134, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__26 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__26, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x28x28xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(batch_norm__138, full_int_array_19, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [2, 3]

        # pd_op.squeeze_: (-1x512xf32, None) <- (-1x512x1x1xf32, 2xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_7, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x32xf32) <- (-1x512xf32, 512x32xf32)
        matmul_12 = paddle._C_ops.matmul(squeeze__12, parameter_144, False, False)

        # pd_op.add_: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add__18 = paddle._C_ops.add_(matmul_12, parameter_145)

        # pd_op.relu_: (-1x32xf32) <- (-1x32xf32)
        relu__27 = paddle._C_ops.relu_(add__18)

        # pd_op.matmul: (-1x512xf32) <- (-1x32xf32, 32x512xf32)
        matmul_13 = paddle._C_ops.matmul(relu__27, parameter_146, False, False)

        # pd_op.add_: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add__19 = paddle._C_ops.add_(matmul_13, parameter_147)

        # pd_op.sigmoid_: (-1x512xf32) <- (-1x512xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(add__19)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [2, 3]

        # pd_op.unsqueeze_: (-1x512x1x1xf32, None) <- (-1x512xf32, 2xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__6, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(batch_norm__138, unsqueeze__12)

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__20 = paddle._C_ops.add_(relu__24, multiply__6)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__28 = paddle._C_ops.relu_(add__20)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x512x28x28xf32, 512x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__28, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_149, parameter_150, parameter_151, parameter_152, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__29 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x28x28xf32, 512x16x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__29, parameter_153, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_154, parameter_155, parameter_156, parameter_157, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__30 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__30, parameter_158, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_159, parameter_160, parameter_161, parameter_162, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(batch_norm__156, full_int_array_22, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_8, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_14 = paddle._C_ops.matmul(squeeze__14, parameter_163, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__21 = paddle._C_ops.add_(matmul_14, parameter_164)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__31 = paddle._C_ops.relu_(add__21)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_15 = paddle._C_ops.matmul(relu__31, parameter_165, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__22 = paddle._C_ops.add_(matmul_15, parameter_166)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(add__22)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__7, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__7 = paddle._C_ops.multiply_(batch_norm__156, unsqueeze__14)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x28x28xf32, 1024x512x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__28, parameter_167, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__23 = paddle._C_ops.add_(batch_norm__162, multiply__7)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__32 = paddle._C_ops.relu_(add__23)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__32, parameter_172, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_173, parameter_174, parameter_175, parameter_176, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__33 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__33, parameter_177, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__34 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__34, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_183, parameter_184, parameter_185, parameter_186, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(batch_norm__180, full_int_array_25, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_9, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_16 = paddle._C_ops.matmul(squeeze__16, parameter_187, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__24 = paddle._C_ops.add_(matmul_16, parameter_188)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__35 = paddle._C_ops.relu_(add__24)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_17 = paddle._C_ops.matmul(relu__35, parameter_189, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__25 = paddle._C_ops.add_(matmul_17, parameter_190)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(add__25)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_27 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__8, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__8 = paddle._C_ops.multiply_(batch_norm__180, unsqueeze__16)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__26 = paddle._C_ops.add_(relu__32, multiply__8)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__36 = paddle._C_ops.relu_(add__26)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__36, parameter_191, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_192, parameter_193, parameter_194, parameter_195, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__37 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__37, parameter_196, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__38 = paddle._C_ops.relu_(batch_norm__192)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu__38, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(batch_norm__198, full_int_array_28, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_29 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_10, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_18 = paddle._C_ops.matmul(squeeze__18, parameter_206, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__27 = paddle._C_ops.add_(matmul_18, parameter_207)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__39 = paddle._C_ops.relu_(add__27)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_19 = paddle._C_ops.matmul(relu__39, parameter_208, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__28 = paddle._C_ops.add_(matmul_19, parameter_209)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__9 = paddle._C_ops.sigmoid_(add__28)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_30 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__9, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__9 = paddle._C_ops.multiply_(batch_norm__198, unsqueeze__18)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__29 = paddle._C_ops.add_(relu__36, multiply__9)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__40 = paddle._C_ops.relu_(add__29)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__40, parameter_210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__41 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__41, parameter_215, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__42 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__42, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(batch_norm__216, full_int_array_31, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__20, squeeze__21 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_11, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_20 = paddle._C_ops.matmul(squeeze__20, parameter_225, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__30 = paddle._C_ops.add_(matmul_20, parameter_226)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__43 = paddle._C_ops.relu_(add__30)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_21 = paddle._C_ops.matmul(relu__43, parameter_227, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__31 = paddle._C_ops.add_(matmul_21, parameter_228)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__10 = paddle._C_ops.sigmoid_(add__31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_33 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__20, unsqueeze__21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__10, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__10 = paddle._C_ops.multiply_(batch_norm__216, unsqueeze__20)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__32 = paddle._C_ops.add_(relu__40, multiply__10)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__44 = paddle._C_ops.relu_(add__32)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu__44, parameter_229, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__45 = paddle._C_ops.relu_(batch_norm__222)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__45, parameter_234, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__46 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu__46, parameter_239, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_240, parameter_241, parameter_242, parameter_243, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_34 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(batch_norm__234, full_int_array_34, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_35 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__22, squeeze__23 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_12, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_22 = paddle._C_ops.matmul(squeeze__22, parameter_244, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__33 = paddle._C_ops.add_(matmul_22, parameter_245)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__47 = paddle._C_ops.relu_(add__33)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_23 = paddle._C_ops.matmul(relu__47, parameter_246, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__34 = paddle._C_ops.add_(matmul_23, parameter_247)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__11 = paddle._C_ops.sigmoid_(add__34)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__22, unsqueeze__23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__11, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__11 = paddle._C_ops.multiply_(batch_norm__234, unsqueeze__22)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__35 = paddle._C_ops.add_(relu__44, multiply__11)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__48 = paddle._C_ops.relu_(add__35)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__48, parameter_248, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_249, parameter_250, parameter_251, parameter_252, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__49 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu__49, parameter_253, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_254, parameter_255, parameter_256, parameter_257, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__50 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu__50, parameter_258, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_259, parameter_260, parameter_261, parameter_262, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(batch_norm__252, full_int_array_37, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__24, squeeze__25 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_13, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_24 = paddle._C_ops.matmul(squeeze__24, parameter_263, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__36 = paddle._C_ops.add_(matmul_24, parameter_264)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__51 = paddle._C_ops.relu_(add__36)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_25 = paddle._C_ops.matmul(relu__51, parameter_265, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__37 = paddle._C_ops.add_(matmul_25, parameter_266)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__12 = paddle._C_ops.sigmoid_(add__37)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__24, unsqueeze__25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__12, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__12 = paddle._C_ops.multiply_(batch_norm__252, unsqueeze__24)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__38 = paddle._C_ops.add_(relu__48, multiply__12)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__52 = paddle._C_ops.relu_(add__38)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__52, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__53 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__53, parameter_272, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__54 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu__54, parameter_277, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_278, parameter_279, parameter_280, parameter_281, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(batch_norm__270, full_int_array_40, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__26, squeeze__27 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_14, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_26 = paddle._C_ops.matmul(squeeze__26, parameter_282, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__39 = paddle._C_ops.add_(matmul_26, parameter_283)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__55 = paddle._C_ops.relu_(add__39)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_27 = paddle._C_ops.matmul(relu__55, parameter_284, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__40 = paddle._C_ops.add_(matmul_27, parameter_285)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__13 = paddle._C_ops.sigmoid_(add__40)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__26, unsqueeze__27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__13, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__13 = paddle._C_ops.multiply_(batch_norm__270, unsqueeze__26)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__41 = paddle._C_ops.add_(relu__52, multiply__13)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__56 = paddle._C_ops.relu_(add__41)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu__56, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_287, parameter_288, parameter_289, parameter_290, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__57 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__57, parameter_291, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_292, parameter_293, parameter_294, parameter_295, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__58 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__58, parameter_296, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(batch_norm__288, full_int_array_43, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__28, squeeze__29 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_15, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_28 = paddle._C_ops.matmul(squeeze__28, parameter_301, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__42 = paddle._C_ops.add_(matmul_28, parameter_302)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__59 = paddle._C_ops.relu_(add__42)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_29 = paddle._C_ops.matmul(relu__59, parameter_303, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__43 = paddle._C_ops.add_(matmul_29, parameter_304)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__14 = paddle._C_ops.sigmoid_(add__43)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_45 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__28, unsqueeze__29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__14, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__14 = paddle._C_ops.multiply_(batch_norm__288, unsqueeze__28)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__44 = paddle._C_ops.add_(relu__56, multiply__14)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__60 = paddle._C_ops.relu_(add__44)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__60, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__61 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu__61, parameter_310, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__62 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__62, parameter_315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_46 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_16 = paddle._C_ops.pool2d(batch_norm__306, full_int_array_46, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_47 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__30, squeeze__31 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_16, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_30 = paddle._C_ops.matmul(squeeze__30, parameter_320, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__45 = paddle._C_ops.add_(matmul_30, parameter_321)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__63 = paddle._C_ops.relu_(add__45)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_31 = paddle._C_ops.matmul(relu__63, parameter_322, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__46 = paddle._C_ops.add_(matmul_31, parameter_323)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__15 = paddle._C_ops.sigmoid_(add__46)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_48 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__30, unsqueeze__31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__15, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__15 = paddle._C_ops.multiply_(batch_norm__306, unsqueeze__30)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__47 = paddle._C_ops.add_(relu__60, multiply__15)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__64 = paddle._C_ops.relu_(add__47)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__64, parameter_324, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_325, parameter_326, parameter_327, parameter_328, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__65 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__65, parameter_329, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_330, parameter_331, parameter_332, parameter_333, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__66 = paddle._C_ops.relu_(batch_norm__318)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__66, parameter_334, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_335, parameter_336, parameter_337, parameter_338, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_49 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_17 = paddle._C_ops.pool2d(batch_norm__324, full_int_array_49, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__32, squeeze__33 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_17, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_32 = paddle._C_ops.matmul(squeeze__32, parameter_339, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__48 = paddle._C_ops.add_(matmul_32, parameter_340)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__67 = paddle._C_ops.relu_(add__48)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_33 = paddle._C_ops.matmul(relu__67, parameter_341, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__49 = paddle._C_ops.add_(matmul_33, parameter_342)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__16 = paddle._C_ops.sigmoid_(add__49)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_51 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__32, unsqueeze__33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__16, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__16 = paddle._C_ops.multiply_(batch_norm__324, unsqueeze__32)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__50 = paddle._C_ops.add_(relu__64, multiply__16)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__68 = paddle._C_ops.relu_(add__50)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(relu__68, parameter_343, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_344, parameter_345, parameter_346, parameter_347, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__69 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu__69, parameter_348, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_349, parameter_350, parameter_351, parameter_352, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__70 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu__70, parameter_353, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_354, parameter_355, parameter_356, parameter_357, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_52 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_18 = paddle._C_ops.pool2d(batch_norm__342, full_int_array_52, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_53 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__34, squeeze__35 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_18, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_34 = paddle._C_ops.matmul(squeeze__34, parameter_358, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__51 = paddle._C_ops.add_(matmul_34, parameter_359)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__71 = paddle._C_ops.relu_(add__51)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_35 = paddle._C_ops.matmul(relu__71, parameter_360, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__52 = paddle._C_ops.add_(matmul_35, parameter_361)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__17 = paddle._C_ops.sigmoid_(add__52)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_54 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__34, unsqueeze__35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__17, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__17 = paddle._C_ops.multiply_(batch_norm__342, unsqueeze__34)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__53 = paddle._C_ops.add_(relu__68, multiply__17)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__72 = paddle._C_ops.relu_(add__53)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(relu__72, parameter_362, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_363, parameter_364, parameter_365, parameter_366, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__73 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(relu__73, parameter_367, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_368, parameter_369, parameter_370, parameter_371, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__74 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(relu__74, parameter_372, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_373, parameter_374, parameter_375, parameter_376, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_55 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_19 = paddle._C_ops.pool2d(batch_norm__360, full_int_array_55, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_56 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__36, squeeze__37 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_19, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_36 = paddle._C_ops.matmul(squeeze__36, parameter_377, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__54 = paddle._C_ops.add_(matmul_36, parameter_378)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__75 = paddle._C_ops.relu_(add__54)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_37 = paddle._C_ops.matmul(relu__75, parameter_379, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__55 = paddle._C_ops.add_(matmul_37, parameter_380)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__18 = paddle._C_ops.sigmoid_(add__55)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_57 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__36, unsqueeze__37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__18, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__18 = paddle._C_ops.multiply_(batch_norm__360, unsqueeze__36)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__56 = paddle._C_ops.add_(relu__72, multiply__18)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__76 = paddle._C_ops.relu_(add__56)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(relu__76, parameter_381, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_382, parameter_383, parameter_384, parameter_385, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__77 = paddle._C_ops.relu_(batch_norm__366)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu__77, parameter_386, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_387, parameter_388, parameter_389, parameter_390, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__78 = paddle._C_ops.relu_(batch_norm__372)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(relu__78, parameter_391, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_392, parameter_393, parameter_394, parameter_395, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_20 = paddle._C_ops.pool2d(batch_norm__378, full_int_array_58, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_59 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__38, squeeze__39 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_20, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_38 = paddle._C_ops.matmul(squeeze__38, parameter_396, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__57 = paddle._C_ops.add_(matmul_38, parameter_397)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__79 = paddle._C_ops.relu_(add__57)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_39 = paddle._C_ops.matmul(relu__79, parameter_398, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__58 = paddle._C_ops.add_(matmul_39, parameter_399)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__19 = paddle._C_ops.sigmoid_(add__58)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_60 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__38, unsqueeze__39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__19, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__19 = paddle._C_ops.multiply_(batch_norm__378, unsqueeze__38)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__59 = paddle._C_ops.add_(relu__76, multiply__19)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__80 = paddle._C_ops.relu_(add__59)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(relu__80, parameter_400, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_401, parameter_402, parameter_403, parameter_404, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__81 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_65 = paddle._C_ops.conv2d(relu__81, parameter_405, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_406, parameter_407, parameter_408, parameter_409, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__82 = paddle._C_ops.relu_(batch_norm__390)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(relu__82, parameter_410, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_61 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_21 = paddle._C_ops.pool2d(batch_norm__396, full_int_array_61, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_62 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__40, squeeze__41 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_21, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_40 = paddle._C_ops.matmul(squeeze__40, parameter_415, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__60 = paddle._C_ops.add_(matmul_40, parameter_416)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__83 = paddle._C_ops.relu_(add__60)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_41 = paddle._C_ops.matmul(relu__83, parameter_417, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__61 = paddle._C_ops.add_(matmul_41, parameter_418)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__20 = paddle._C_ops.sigmoid_(add__61)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_63 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__40, unsqueeze__41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__20, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__20 = paddle._C_ops.multiply_(batch_norm__396, unsqueeze__40)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__62 = paddle._C_ops.add_(relu__80, multiply__20)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__84 = paddle._C_ops.relu_(add__62)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu__84, parameter_419, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_420, parameter_421, parameter_422, parameter_423, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__85 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_68 = paddle._C_ops.conv2d(relu__85, parameter_424, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_425, parameter_426, parameter_427, parameter_428, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__86 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu__86, parameter_429, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_430, parameter_431, parameter_432, parameter_433, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_64 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_22 = paddle._C_ops.pool2d(batch_norm__414, full_int_array_64, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_65 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__42, squeeze__43 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_22, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_42 = paddle._C_ops.matmul(squeeze__42, parameter_434, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__63 = paddle._C_ops.add_(matmul_42, parameter_435)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__87 = paddle._C_ops.relu_(add__63)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_43 = paddle._C_ops.matmul(relu__87, parameter_436, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__64 = paddle._C_ops.add_(matmul_43, parameter_437)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__21 = paddle._C_ops.sigmoid_(add__64)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_66 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__42, unsqueeze__43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__21, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__21 = paddle._C_ops.multiply_(batch_norm__414, unsqueeze__42)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__65 = paddle._C_ops.add_(relu__84, multiply__21)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__88 = paddle._C_ops.relu_(add__65)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(relu__88, parameter_438, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_439, parameter_440, parameter_441, parameter_442, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__89 = paddle._C_ops.relu_(batch_norm__420)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_71 = paddle._C_ops.conv2d(relu__89, parameter_443, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_444, parameter_445, parameter_446, parameter_447, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__90 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(relu__90, parameter_448, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_449, parameter_450, parameter_451, parameter_452, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_67 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_23 = paddle._C_ops.pool2d(batch_norm__432, full_int_array_67, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_68 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__44, squeeze__45 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_23, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_44 = paddle._C_ops.matmul(squeeze__44, parameter_453, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__66 = paddle._C_ops.add_(matmul_44, parameter_454)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__91 = paddle._C_ops.relu_(add__66)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_45 = paddle._C_ops.matmul(relu__91, parameter_455, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__67 = paddle._C_ops.add_(matmul_45, parameter_456)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__22 = paddle._C_ops.sigmoid_(add__67)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_69 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__44, unsqueeze__45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__22, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__22 = paddle._C_ops.multiply_(batch_norm__432, unsqueeze__44)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__68 = paddle._C_ops.add_(relu__88, multiply__22)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__92 = paddle._C_ops.relu_(add__68)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(relu__92, parameter_457, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_458, parameter_459, parameter_460, parameter_461, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__93 = paddle._C_ops.relu_(batch_norm__438)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu__93, parameter_462, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_463, parameter_464, parameter_465, parameter_466, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__94 = paddle._C_ops.relu_(batch_norm__444)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(relu__94, parameter_467, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_468, parameter_469, parameter_470, parameter_471, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_70 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_24 = paddle._C_ops.pool2d(batch_norm__450, full_int_array_70, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_71 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__46, squeeze__47 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_24, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_46 = paddle._C_ops.matmul(squeeze__46, parameter_472, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__69 = paddle._C_ops.add_(matmul_46, parameter_473)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__95 = paddle._C_ops.relu_(add__69)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_47 = paddle._C_ops.matmul(relu__95, parameter_474, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__70 = paddle._C_ops.add_(matmul_47, parameter_475)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__23 = paddle._C_ops.sigmoid_(add__70)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_72 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__46, unsqueeze__47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__23, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__23 = paddle._C_ops.multiply_(batch_norm__450, unsqueeze__46)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__71 = paddle._C_ops.add_(relu__92, multiply__23)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__96 = paddle._C_ops.relu_(add__71)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__96, parameter_476, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_477, parameter_478, parameter_479, parameter_480, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__97 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu__97, parameter_481, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_482, parameter_483, parameter_484, parameter_485, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__98 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(relu__98, parameter_486, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_487, parameter_488, parameter_489, parameter_490, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_73 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_25 = paddle._C_ops.pool2d(batch_norm__468, full_int_array_73, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_74 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__48, squeeze__49 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_25, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_48 = paddle._C_ops.matmul(squeeze__48, parameter_491, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__72 = paddle._C_ops.add_(matmul_48, parameter_492)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__99 = paddle._C_ops.relu_(add__72)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_49 = paddle._C_ops.matmul(relu__99, parameter_493, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__73 = paddle._C_ops.add_(matmul_49, parameter_494)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__24 = paddle._C_ops.sigmoid_(add__73)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_75 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__48, unsqueeze__49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__24, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__24 = paddle._C_ops.multiply_(batch_norm__468, unsqueeze__48)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__74 = paddle._C_ops.add_(relu__96, multiply__24)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__100 = paddle._C_ops.relu_(add__74)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu__100, parameter_495, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_496, parameter_497, parameter_498, parameter_499, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__101 = paddle._C_ops.relu_(batch_norm__474)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu__101, parameter_500, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_501, parameter_502, parameter_503, parameter_504, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__102 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(relu__102, parameter_505, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_506, parameter_507, parameter_508, parameter_509, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_76 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_26 = paddle._C_ops.pool2d(batch_norm__486, full_int_array_76, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_77 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__50, squeeze__51 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_26, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_50 = paddle._C_ops.matmul(squeeze__50, parameter_510, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__75 = paddle._C_ops.add_(matmul_50, parameter_511)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__103 = paddle._C_ops.relu_(add__75)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_51 = paddle._C_ops.matmul(relu__103, parameter_512, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__76 = paddle._C_ops.add_(matmul_51, parameter_513)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__25 = paddle._C_ops.sigmoid_(add__76)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_78 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__50, unsqueeze__51 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__25, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__25 = paddle._C_ops.multiply_(batch_norm__486, unsqueeze__50)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__77 = paddle._C_ops.add_(relu__100, multiply__25)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__104 = paddle._C_ops.relu_(add__77)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu__104, parameter_514, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_515, parameter_516, parameter_517, parameter_518, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__105 = paddle._C_ops.relu_(batch_norm__492)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu__105, parameter_519, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_520, parameter_521, parameter_522, parameter_523, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__106 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu__106, parameter_524, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_525, parameter_526, parameter_527, parameter_528, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_79 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_27 = paddle._C_ops.pool2d(batch_norm__504, full_int_array_79, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_80 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__52, squeeze__53 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_27, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_52 = paddle._C_ops.matmul(squeeze__52, parameter_529, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__78 = paddle._C_ops.add_(matmul_52, parameter_530)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__107 = paddle._C_ops.relu_(add__78)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_53 = paddle._C_ops.matmul(relu__107, parameter_531, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__79 = paddle._C_ops.add_(matmul_53, parameter_532)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__26 = paddle._C_ops.sigmoid_(add__79)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_81 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__52, unsqueeze__53 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__26, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__26 = paddle._C_ops.multiply_(batch_norm__504, unsqueeze__52)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__80 = paddle._C_ops.add_(relu__104, multiply__26)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__108 = paddle._C_ops.relu_(add__80)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu__108, parameter_533, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_534, parameter_535, parameter_536, parameter_537, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__109 = paddle._C_ops.relu_(batch_norm__510)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_86 = paddle._C_ops.conv2d(relu__109, parameter_538, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_539, parameter_540, parameter_541, parameter_542, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__110 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(relu__110, parameter_543, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_544, parameter_545, parameter_546, parameter_547, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_82 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_28 = paddle._C_ops.pool2d(batch_norm__522, full_int_array_82, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_83 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__54, squeeze__55 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_28, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_54 = paddle._C_ops.matmul(squeeze__54, parameter_548, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__81 = paddle._C_ops.add_(matmul_54, parameter_549)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__111 = paddle._C_ops.relu_(add__81)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_55 = paddle._C_ops.matmul(relu__111, parameter_550, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__82 = paddle._C_ops.add_(matmul_55, parameter_551)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__27 = paddle._C_ops.sigmoid_(add__82)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_84 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__54, unsqueeze__55 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__27, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__27 = paddle._C_ops.multiply_(batch_norm__522, unsqueeze__54)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__83 = paddle._C_ops.add_(relu__108, multiply__27)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__112 = paddle._C_ops.relu_(add__83)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(relu__112, parameter_552, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_553, parameter_554, parameter_555, parameter_556, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__113 = paddle._C_ops.relu_(batch_norm__528)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_89 = paddle._C_ops.conv2d(relu__113, parameter_557, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_558, parameter_559, parameter_560, parameter_561, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__114 = paddle._C_ops.relu_(batch_norm__534)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(relu__114, parameter_562, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_563, parameter_564, parameter_565, parameter_566, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_85 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_29 = paddle._C_ops.pool2d(batch_norm__540, full_int_array_85, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_86 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__56, squeeze__57 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_29, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_56 = paddle._C_ops.matmul(squeeze__56, parameter_567, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__84 = paddle._C_ops.add_(matmul_56, parameter_568)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__115 = paddle._C_ops.relu_(add__84)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_57 = paddle._C_ops.matmul(relu__115, parameter_569, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__85 = paddle._C_ops.add_(matmul_57, parameter_570)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__28 = paddle._C_ops.sigmoid_(add__85)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_87 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__56, unsqueeze__57 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__28, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__28 = paddle._C_ops.multiply_(batch_norm__540, unsqueeze__56)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__86 = paddle._C_ops.add_(relu__112, multiply__28)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__116 = paddle._C_ops.relu_(add__86)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(relu__116, parameter_571, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_572, parameter_573, parameter_574, parameter_575, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__117 = paddle._C_ops.relu_(batch_norm__546)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 512x16x3x3xf32)
        conv2d_92 = paddle._C_ops.conv2d(relu__117, parameter_576, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_577, parameter_578, parameter_579, parameter_580, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__118 = paddle._C_ops.relu_(batch_norm__552)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(relu__118, parameter_581, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_582, parameter_583, parameter_584, parameter_585, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_88 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x14x14xf32, 2xi64)
        pool2d_30 = paddle._C_ops.pool2d(batch_norm__558, full_int_array_88, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_89 = [2, 3]

        # pd_op.squeeze_: (-1x1024xf32, None) <- (-1x1024x1x1xf32, 2xi64)
        squeeze__58, squeeze__59 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_30, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64xf32) <- (-1x1024xf32, 1024x64xf32)
        matmul_58 = paddle._C_ops.matmul(squeeze__58, parameter_586, False, False)

        # pd_op.add_: (-1x64xf32) <- (-1x64xf32, 64xf32)
        add__87 = paddle._C_ops.add_(matmul_58, parameter_587)

        # pd_op.relu_: (-1x64xf32) <- (-1x64xf32)
        relu__119 = paddle._C_ops.relu_(add__87)

        # pd_op.matmul: (-1x1024xf32) <- (-1x64xf32, 64x1024xf32)
        matmul_59 = paddle._C_ops.matmul(relu__119, parameter_588, False, False)

        # pd_op.add_: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add__88 = paddle._C_ops.add_(matmul_59, parameter_589)

        # pd_op.sigmoid_: (-1x1024xf32) <- (-1x1024xf32)
        sigmoid__29 = paddle._C_ops.sigmoid_(add__88)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_90 = [2, 3]

        # pd_op.unsqueeze_: (-1x1024x1x1xf32, None) <- (-1x1024xf32, 2xi64)
        unsqueeze__58, unsqueeze__59 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__29, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x1x1xf32)
        multiply__29 = paddle._C_ops.multiply_(batch_norm__558, unsqueeze__58)

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__89 = paddle._C_ops.add_(relu__116, multiply__29)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__120 = paddle._C_ops.relu_(add__89)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, 1024x1024x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu__120, parameter_590, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_591, parameter_592, parameter_593, parameter_594, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__121 = paddle._C_ops.relu_(batch_norm__564)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x1024x14x14xf32, 1024x32x3x3xf32)
        conv2d_95 = paddle._C_ops.conv2d(relu__121, parameter_595, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_596, parameter_597, parameter_598, parameter_599, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu__122 = paddle._C_ops.relu_(batch_norm__570)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu__122, parameter_600, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_601, parameter_602, parameter_603, parameter_604, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_91 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        pool2d_31 = paddle._C_ops.pool2d(batch_norm__576, full_int_array_91, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_92 = [2, 3]

        # pd_op.squeeze_: (-1x2048xf32, None) <- (-1x2048x1x1xf32, 2xi64)
        squeeze__60, squeeze__61 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_31, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x128xf32) <- (-1x2048xf32, 2048x128xf32)
        matmul_60 = paddle._C_ops.matmul(squeeze__60, parameter_605, False, False)

        # pd_op.add_: (-1x128xf32) <- (-1x128xf32, 128xf32)
        add__90 = paddle._C_ops.add_(matmul_60, parameter_606)

        # pd_op.relu_: (-1x128xf32) <- (-1x128xf32)
        relu__123 = paddle._C_ops.relu_(add__90)

        # pd_op.matmul: (-1x2048xf32) <- (-1x128xf32, 128x2048xf32)
        matmul_61 = paddle._C_ops.matmul(relu__123, parameter_607, False, False)

        # pd_op.add_: (-1x2048xf32) <- (-1x2048xf32, 2048xf32)
        add__91 = paddle._C_ops.add_(matmul_61, parameter_608)

        # pd_op.sigmoid_: (-1x2048xf32) <- (-1x2048xf32)
        sigmoid__30 = paddle._C_ops.sigmoid_(add__91)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_93 = [2, 3]

        # pd_op.unsqueeze_: (-1x2048x1x1xf32, None) <- (-1x2048xf32, 2xi64)
        unsqueeze__60, unsqueeze__61 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__30, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x1x1xf32)
        multiply__30 = paddle._C_ops.multiply_(batch_norm__576, unsqueeze__60)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x14x14xf32, 2048x1024x1x1xf32)
        conv2d_97 = paddle._C_ops.conv2d(relu__120, parameter_609, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_610, parameter_611, parameter_612, parameter_613, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__92 = paddle._C_ops.add_(batch_norm__582, multiply__30)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__124 = paddle._C_ops.relu_(add__92)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_98 = paddle._C_ops.conv2d(relu__124, parameter_614, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_615, parameter_616, parameter_617, parameter_618, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu__125 = paddle._C_ops.relu_(batch_norm__588)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x32x3x3xf32)
        conv2d_99 = paddle._C_ops.conv2d(relu__125, parameter_619, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_620, parameter_621, parameter_622, parameter_623, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu__126 = paddle._C_ops.relu_(batch_norm__594)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_100 = paddle._C_ops.conv2d(relu__126, parameter_624, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_625, parameter_626, parameter_627, parameter_628, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_94 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        pool2d_32 = paddle._C_ops.pool2d(batch_norm__600, full_int_array_94, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_95 = [2, 3]

        # pd_op.squeeze_: (-1x2048xf32, None) <- (-1x2048x1x1xf32, 2xi64)
        squeeze__62, squeeze__63 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_32, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x128xf32) <- (-1x2048xf32, 2048x128xf32)
        matmul_62 = paddle._C_ops.matmul(squeeze__62, parameter_629, False, False)

        # pd_op.add_: (-1x128xf32) <- (-1x128xf32, 128xf32)
        add__93 = paddle._C_ops.add_(matmul_62, parameter_630)

        # pd_op.relu_: (-1x128xf32) <- (-1x128xf32)
        relu__127 = paddle._C_ops.relu_(add__93)

        # pd_op.matmul: (-1x2048xf32) <- (-1x128xf32, 128x2048xf32)
        matmul_63 = paddle._C_ops.matmul(relu__127, parameter_631, False, False)

        # pd_op.add_: (-1x2048xf32) <- (-1x2048xf32, 2048xf32)
        add__94 = paddle._C_ops.add_(matmul_63, parameter_632)

        # pd_op.sigmoid_: (-1x2048xf32) <- (-1x2048xf32)
        sigmoid__31 = paddle._C_ops.sigmoid_(add__94)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_96 = [2, 3]

        # pd_op.unsqueeze_: (-1x2048x1x1xf32, None) <- (-1x2048xf32, 2xi64)
        unsqueeze__62, unsqueeze__63 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__31, full_int_array_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x1x1xf32)
        multiply__31 = paddle._C_ops.multiply_(batch_norm__600, unsqueeze__62)

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__95 = paddle._C_ops.add_(relu__124, multiply__31)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__128 = paddle._C_ops.relu_(add__95)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu__128, parameter_633, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_634, parameter_635, parameter_636, parameter_637, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu__129 = paddle._C_ops.relu_(batch_norm__606)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1024x32x3x3xf32)
        conv2d_102 = paddle._C_ops.conv2d(relu__129, parameter_638, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_639, parameter_640, parameter_641, parameter_642, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu__130 = paddle._C_ops.relu_(batch_norm__612)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_103 = paddle._C_ops.conv2d(relu__130, parameter_643, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_644, parameter_645, parameter_646, parameter_647, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_97 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        pool2d_33 = paddle._C_ops.pool2d(batch_norm__618, full_int_array_97, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_98 = [2, 3]

        # pd_op.squeeze_: (-1x2048xf32, None) <- (-1x2048x1x1xf32, 2xi64)
        squeeze__64, squeeze__65 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_33, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x128xf32) <- (-1x2048xf32, 2048x128xf32)
        matmul_64 = paddle._C_ops.matmul(squeeze__64, parameter_648, False, False)

        # pd_op.add_: (-1x128xf32) <- (-1x128xf32, 128xf32)
        add__96 = paddle._C_ops.add_(matmul_64, parameter_649)

        # pd_op.relu_: (-1x128xf32) <- (-1x128xf32)
        relu__131 = paddle._C_ops.relu_(add__96)

        # pd_op.matmul: (-1x2048xf32) <- (-1x128xf32, 128x2048xf32)
        matmul_65 = paddle._C_ops.matmul(relu__131, parameter_650, False, False)

        # pd_op.add_: (-1x2048xf32) <- (-1x2048xf32, 2048xf32)
        add__97 = paddle._C_ops.add_(matmul_65, parameter_651)

        # pd_op.sigmoid_: (-1x2048xf32) <- (-1x2048xf32)
        sigmoid__32 = paddle._C_ops.sigmoid_(add__97)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_99 = [2, 3]

        # pd_op.unsqueeze_: (-1x2048x1x1xf32, None) <- (-1x2048xf32, 2xi64)
        unsqueeze__64, unsqueeze__65 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(sigmoid__32, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x1x1xf32)
        multiply__32 = paddle._C_ops.multiply_(batch_norm__618, unsqueeze__64)

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__98 = paddle._C_ops.add_(relu__128, multiply__32)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__132 = paddle._C_ops.relu_(add__98)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_100 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        pool2d_34 = paddle._C_ops.pool2d(relu__132, full_int_array_100, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_101 = [-1, 2048]

        # pd_op.reshape_: (-1x2048xf32, 0x-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 2xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_34, full_int_array_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x2048xf32, 2048x1000xf32)
        matmul_66 = paddle._C_ops.matmul(reshape__0, parameter_652, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__99 = paddle._C_ops.add_(matmul_66, parameter_653)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax_(add__99, -1)
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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_341, parameter_342, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_417, parameter_418, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_436, parameter_437, parameter_438, parameter_442, parameter_439, parameter_441, parameter_440, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_454, parameter_455, parameter_456, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_473, parameter_474, parameter_475, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_566, parameter_563, parameter_565, parameter_564, parameter_567, parameter_568, parameter_569, parameter_570, parameter_571, parameter_575, parameter_572, parameter_574, parameter_573, parameter_576, parameter_580, parameter_577, parameter_579, parameter_578, parameter_581, parameter_585, parameter_582, parameter_584, parameter_583, parameter_586, parameter_587, parameter_588, parameter_589, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_606, parameter_607, parameter_608, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_630, parameter_631, parameter_632, parameter_633, parameter_637, parameter_634, parameter_636, parameter_635, parameter_638, parameter_642, parameter_639, parameter_641, parameter_640, parameter_643, parameter_647, parameter_644, parameter_646, parameter_645, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_653, feed_0):
        return self.builtin_module_1440_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_52, parameter_49, parameter_51, parameter_50, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_152, parameter_149, parameter_151, parameter_150, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_323, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_341, parameter_342, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_379, parameter_380, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_398, parameter_399, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_417, parameter_418, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_436, parameter_437, parameter_438, parameter_442, parameter_439, parameter_441, parameter_440, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_454, parameter_455, parameter_456, parameter_457, parameter_461, parameter_458, parameter_460, parameter_459, parameter_462, parameter_466, parameter_463, parameter_465, parameter_464, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_473, parameter_474, parameter_475, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_566, parameter_563, parameter_565, parameter_564, parameter_567, parameter_568, parameter_569, parameter_570, parameter_571, parameter_575, parameter_572, parameter_574, parameter_573, parameter_576, parameter_580, parameter_577, parameter_579, parameter_578, parameter_581, parameter_585, parameter_582, parameter_584, parameter_583, parameter_586, parameter_587, parameter_588, parameter_589, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_606, parameter_607, parameter_608, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_630, parameter_631, parameter_632, parameter_633, parameter_637, parameter_634, parameter_636, parameter_635, parameter_638, parameter_642, parameter_639, parameter_641, parameter_640, parameter_643, parameter_647, parameter_644, parameter_646, parameter_645, parameter_648, parameter_649, parameter_650, parameter_651, parameter_652, parameter_653, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1440_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128, 4, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 16], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([16, 256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128, 4, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256, 16], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([16, 256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128, 4, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256, 16], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([16, 256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512, 32], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([32, 512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256, 8, 3, 3], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512, 32], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([32, 512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([256, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 32], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([32, 512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([256, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512, 32], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([32, 512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([512, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([1024, 64], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([64, 1024], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([1024, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([1024, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_604
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([2048, 128], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([128, 2048], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_609
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([1024, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_619
            paddle.uniform([1024, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_624
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_628
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_629
            paddle.uniform([2048, 128], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([128, 2048], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([1024, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_634
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([1024, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_644
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([2048, 128], dtype='float32', min=0, max=0.5),
            # parameter_649
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([128, 2048], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            # parameter_653
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
            paddle.static.InputSpec(shape=[64, 3, 7, 7], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 16], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[16, 256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256, 16], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[16, 256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256, 16], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[16, 256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512, 32], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[32, 512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float32'),
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
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512, 32], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[32, 512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 32], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[32, 512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512, 32], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[32, 512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[1024, 64], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[64, 1024], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_604
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[2048, 128], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[128, 2048], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_609
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[1024, 2048, 1, 1], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_619
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_624
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_628
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_629
            paddle.static.InputSpec(shape=[2048, 128], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[128, 2048], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[1024, 2048, 1, 1], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_634
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_644
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[2048, 128], dtype='float32'),
            # parameter_649
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[128, 2048], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            # parameter_653
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