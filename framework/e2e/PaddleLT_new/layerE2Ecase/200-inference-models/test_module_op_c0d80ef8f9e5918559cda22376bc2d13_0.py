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
    return [950][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1633_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_368, parameter_365, parameter_367, parameter_366, parameter_369, parameter_373, parameter_370, parameter_372, parameter_371, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_388, parameter_385, parameter_387, parameter_386, parameter_389, parameter_393, parameter_390, parameter_392, parameter_391, parameter_394, parameter_398, parameter_395, parameter_397, parameter_396, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_438, parameter_435, parameter_437, parameter_436, parameter_439, parameter_443, parameter_440, parameter_442, parameter_441, parameter_444, parameter_448, parameter_445, parameter_447, parameter_446, parameter_449, parameter_453, parameter_450, parameter_452, parameter_451, parameter_454, parameter_458, parameter_455, parameter_457, parameter_456, parameter_459, parameter_463, parameter_460, parameter_462, parameter_461, parameter_464, parameter_468, parameter_465, parameter_467, parameter_466, parameter_469, parameter_473, parameter_470, parameter_472, parameter_471, parameter_474, parameter_478, parameter_475, parameter_477, parameter_476, parameter_479, parameter_483, parameter_480, parameter_482, parameter_481, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_498, parameter_495, parameter_497, parameter_496, parameter_499, parameter_503, parameter_500, parameter_502, parameter_501, parameter_504, parameter_508, parameter_505, parameter_507, parameter_506, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_533, parameter_530, parameter_532, parameter_531, parameter_534, parameter_538, parameter_535, parameter_537, parameter_536, parameter_539, parameter_543, parameter_540, parameter_542, parameter_541, parameter_544, parameter_548, parameter_545, parameter_547, parameter_546, parameter_549, parameter_553, parameter_550, parameter_552, parameter_551, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_578, parameter_575, parameter_577, parameter_576, parameter_579, parameter_583, parameter_580, parameter_582, parameter_581, parameter_584, parameter_588, parameter_585, parameter_587, parameter_586, parameter_589, parameter_593, parameter_590, parameter_592, parameter_591, parameter_594, parameter_598, parameter_595, parameter_597, parameter_596, parameter_599, parameter_603, parameter_600, parameter_602, parameter_601, parameter_604, parameter_608, parameter_605, parameter_607, parameter_606, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_643, parameter_640, parameter_642, parameter_641, parameter_644, parameter_648, parameter_645, parameter_647, parameter_646, parameter_649, parameter_653, parameter_650, parameter_652, parameter_651, parameter_654, parameter_658, parameter_655, parameter_657, parameter_656, parameter_659, parameter_663, parameter_660, parameter_662, parameter_661, parameter_664, parameter_668, parameter_665, parameter_667, parameter_666, parameter_669, parameter_673, parameter_670, parameter_672, parameter_671, parameter_674, parameter_675, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x128x112x112xf16) <- (-1x3x224x224xf16, 128x3x7x7xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x112x112xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x112x112xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x112x112xf16) <- (-1x128x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x128x56x56xf16) <- (-1x128x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_0, parameter_5, parameter_6, parameter_7, parameter_8, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x288x56x56xf16) <- (-1x128x56x56xf16, 288x128x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__1, parameter_9, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x56x56xf16, -1x32x56x56xf16]) <- (-1x288x56x56xf16, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(conv2d_1, full_int_array_1, full_0)

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_0, parameter_10, parameter_11, parameter_12, parameter_13, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x128x56x56xf16, 160x128x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__2, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_15, parameter_16, parameter_17, parameter_18, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x160x56x56xf16, 160x4x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__3, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_20, parameter_21, parameter_22, parameter_23, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.conv2d: (-1x272x56x56xf16) <- (-1x160x56x56xf16, 272x160x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__4, parameter_24, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [256, 16]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x56x56xf16, -1x16x56x56xf16]) <- (-1x272x56x56xf16, 2xi64, 1xi32)
        split_1 = paddle._C_ops.split(conv2d_4, full_int_array_2, full_1)

        # builtin.slice: (-1x256x56x56xf16) <- ([-1x256x56x56xf16, -1x32x56x56xf16])
        slice_0 = split_0[0]

        # builtin.slice: (-1x256x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_1 = split_1[0]

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__0 = paddle._C_ops.add_(slice_0, slice_1)

        # builtin.slice: (-1x32x56x56xf16) <- ([-1x256x56x56xf16, -1x32x56x56xf16])
        slice_2 = split_0[1]

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_3 = split_1[1]

        # builtin.combine: ([-1x32x56x56xf16, -1x16x56x56xf16]) <- (-1x32x56x56xf16, -1x16x56x56xf16)
        combine_0 = [slice_2, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x48x56x56xf16) <- ([-1x32x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        # builtin.combine: ([-1x256x56x56xf16, -1x48x56x56xf16]) <- (-1x256x56x56xf16, -1x48x56x56xf16)
        combine_1 = [add__0, concat_0]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x304x56x56xf16) <- ([-1x256x56x56xf16, -1x48x56x56xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_3)

        # pd_op.batch_norm_: (-1x304x56x56xf16, 304xf32, 304xf32, 304xf32, 304xf32, None) <- (-1x304x56x56xf16, 304xf32, 304xf32, 304xf32, 304xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_1, parameter_25, parameter_26, parameter_27, parameter_28, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x304x56x56xf16) <- (-1x304x56x56xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x304x56x56xf16, 160x304x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__5, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x160x56x56xf16, 160x4x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__6, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_35, parameter_36, parameter_37, parameter_38, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x272x56x56xf16) <- (-1x160x56x56xf16, 272x160x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__7, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [256, 16]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x56x56xf16, -1x16x56x56xf16]) <- (-1x272x56x56xf16, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(conv2d_7, full_int_array_3, full_4)

        # builtin.slice: (-1x256x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_4 = split_2[0]

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__1 = paddle._C_ops.add_(add__0, slice_4)

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_5 = split_2[1]

        # builtin.combine: ([-1x48x56x56xf16, -1x16x56x56xf16]) <- (-1x48x56x56xf16, -1x16x56x56xf16)
        combine_2 = [concat_0, slice_5]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x64x56x56xf16) <- ([-1x48x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)

        # builtin.combine: ([-1x256x56x56xf16, -1x64x56x56xf16]) <- (-1x256x56x56xf16, -1x64x56x56xf16)
        combine_3 = [add__1, concat_2]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x56x56xf16) <- ([-1x256x56x56xf16, -1x64x56x56xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_6)

        # pd_op.batch_norm_: (-1x320x56x56xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x56x56xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_3, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x56x56xf16) <- (-1x320x56x56xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x320x56x56xf16, 160x320x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__8, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x160x56x56xf16, 160x4x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__9, parameter_49, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x272x56x56xf16) <- (-1x160x56x56xf16, 272x160x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__10, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [256, 16]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x56x56xf16, -1x16x56x56xf16]) <- (-1x272x56x56xf16, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(conv2d_10, full_int_array_4, full_7)

        # builtin.slice: (-1x256x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_6 = split_3[0]

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__2 = paddle._C_ops.add_(add__1, slice_6)

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_7 = split_3[1]

        # builtin.combine: ([-1x64x56x56xf16, -1x16x56x56xf16]) <- (-1x64x56x56xf16, -1x16x56x56xf16)
        combine_4 = [concat_2, slice_7]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_8)

        # builtin.combine: ([-1x256x56x56xf16, -1x80x56x56xf16]) <- (-1x256x56x56xf16, -1x80x56x56xf16)
        combine_5 = [add__2, concat_4]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x336x56x56xf16) <- ([-1x256x56x56xf16, -1x80x56x56xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_9)

        # pd_op.batch_norm_: (-1x336x56x56xf16, 336xf32, 336xf32, 336xf32, 336xf32, None) <- (-1x336x56x56xf16, 336xf32, 336xf32, 336xf32, 336xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_5, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x336x56x56xf16) <- (-1x336x56x56xf16)
        relu__11 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x336x56x56xf16, 160x336x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__11, parameter_59, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x160x56x56xf16) <- (-1x160x56x56xf16, 160x4x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__12, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32, None) <- (-1x160x56x56xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x56x56xf16) <- (-1x160x56x56xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.conv2d: (-1x272x56x56xf16) <- (-1x160x56x56xf16, 272x160x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__13, parameter_69, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [256, 16]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x56x56xf16, -1x16x56x56xf16]) <- (-1x272x56x56xf16, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(conv2d_13, full_int_array_5, full_10)

        # builtin.slice: (-1x256x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_8 = split_4[0]

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__3 = paddle._C_ops.add_(add__2, slice_8)

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x256x56x56xf16, -1x16x56x56xf16])
        slice_9 = split_4[1]

        # builtin.combine: ([-1x80x56x56xf16, -1x16x56x56xf16]) <- (-1x80x56x56xf16, -1x16x56x56xf16)
        combine_6 = [concat_4, slice_9]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x56x56xf16) <- ([-1x80x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_11)

        # builtin.combine: ([-1x256x56x56xf16, -1x96x56x56xf16]) <- (-1x256x56x56xf16, -1x96x56x56xf16)
        combine_7 = [add__3, concat_6]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x352x56x56xf16) <- ([-1x256x56x56xf16, -1x96x56x56xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_12)

        # pd_op.batch_norm_: (-1x352x56x56xf16, 352xf32, 352xf32, 352xf32, 352xf32, None) <- (-1x352x56x56xf16, 352xf32, 352xf32, 352xf32, 352xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_7, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x352x56x56xf16) <- (-1x352x56x56xf16)
        relu__14 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x576x28x28xf16) <- (-1x352x56x56xf16, 576x352x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__14, parameter_74, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [512, 64]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x64x28x28xf16]) <- (-1x576x28x28xf16, 2xi64, 1xi32)
        split_5 = paddle._C_ops.split(conv2d_14, full_int_array_6, full_13)

        # pd_op.batch_norm_: (-1x352x56x56xf16, 352xf32, 352xf32, 352xf32, 352xf32, None) <- (-1x352x56x56xf16, 352xf32, 352xf32, 352xf32, 352xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_7, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x352x56x56xf16) <- (-1x352x56x56xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x320x56x56xf16) <- (-1x352x56x56xf16, 320x352x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__15, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x56x56xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x56x56xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x56x56xf16) <- (-1x320x56x56xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x56x56xf16, 320x8x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu__16, parameter_84, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__17 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__17, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(conv2d_17, full_int_array_7, full_14)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x64x28x28xf16])
        slice_10 = split_5[0]

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_11 = split_6[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__4 = paddle._C_ops.add_(slice_10, slice_11)

        # builtin.slice: (-1x64x28x28xf16) <- ([-1x512x28x28xf16, -1x64x28x28xf16])
        slice_12 = split_5[1]

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_13 = split_6[1]

        # builtin.combine: ([-1x64x28x28xf16, -1x32x28x28xf16]) <- (-1x64x28x28xf16, -1x32x28x28xf16)
        combine_8 = [slice_12, slice_13]

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x28x28xf16) <- ([-1x64x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_15)

        # builtin.combine: ([-1x512x28x28xf16, -1x96x28x28xf16]) <- (-1x512x28x28xf16, -1x96x28x28xf16)
        combine_9 = [add__4, concat_8]

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x608x28x28xf16) <- ([-1x512x28x28xf16, -1x96x28x28xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_16)

        # pd_op.batch_norm_: (-1x608x28x28xf16, 608xf32, 608xf32, 608xf32, 608xf32, None) <- (-1x608x28x28xf16, 608xf32, 608xf32, 608xf32, 608xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_9, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x608x28x28xf16) <- (-1x608x28x28xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x608x28x28xf16, 320x608x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__18, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__19, parameter_99, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__20, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_7 = paddle._C_ops.split(conv2d_20, full_int_array_8, full_17)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_14 = split_7[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__5 = paddle._C_ops.add_(add__4, slice_14)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_15 = split_7[1]

        # builtin.combine: ([-1x96x28x28xf16, -1x32x28x28xf16]) <- (-1x96x28x28xf16, -1x32x28x28xf16)
        combine_10 = [concat_8, slice_15]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x28x28xf16) <- ([-1x96x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_18)

        # builtin.combine: ([-1x512x28x28xf16, -1x128x28x28xf16]) <- (-1x512x28x28xf16, -1x128x28x28xf16)
        combine_11 = [add__5, concat_10]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x640x28x28xf16) <- ([-1x512x28x28xf16, -1x128x28x28xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_19)

        # pd_op.batch_norm_: (-1x640x28x28xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x28x28xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_11, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x28x28xf16) <- (-1x640x28x28xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x640x28x28xf16, 320x640x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__21, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__22, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__23 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__23, parameter_119, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(conv2d_23, full_int_array_9, full_20)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_16 = split_8[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__6 = paddle._C_ops.add_(add__5, slice_16)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_17 = split_8[1]

        # builtin.combine: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x128x28x28xf16, -1x32x28x28xf16)
        combine_12 = [concat_10, slice_17]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_21)

        # builtin.combine: ([-1x512x28x28xf16, -1x160x28x28xf16]) <- (-1x512x28x28xf16, -1x160x28x28xf16)
        combine_13 = [add__6, concat_12]

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x672x28x28xf16) <- ([-1x512x28x28xf16, -1x160x28x28xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_22)

        # pd_op.batch_norm_: (-1x672x28x28xf16, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x28x28xf16, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_13, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x672x28x28xf16) <- (-1x672x28x28xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x672x28x28xf16, 320x672x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__24, parameter_124, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_125, parameter_126, parameter_127, parameter_128, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__25, parameter_129, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__26, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(conv2d_26, full_int_array_10, full_23)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_18 = split_9[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__7 = paddle._C_ops.add_(add__6, slice_18)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_19 = split_9[1]

        # builtin.combine: ([-1x160x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, -1x32x28x28xf16)
        combine_14 = [concat_12, slice_19]

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x28x28xf16) <- ([-1x160x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_24)

        # builtin.combine: ([-1x512x28x28xf16, -1x192x28x28xf16]) <- (-1x512x28x28xf16, -1x192x28x28xf16)
        combine_15 = [add__7, concat_14]

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x704x28x28xf16) <- ([-1x512x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_25)

        # pd_op.batch_norm_: (-1x704x28x28xf16, 704xf32, 704xf32, 704xf32, 704xf32, None) <- (-1x704x28x28xf16, 704xf32, 704xf32, 704xf32, 704xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_15, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x704x28x28xf16) <- (-1x704x28x28xf16)
        relu__27 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x704x28x28xf16, 320x704x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__27, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__28, parameter_144, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_145, parameter_146, parameter_147, parameter_148, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu__29, parameter_149, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_10 = paddle._C_ops.split(conv2d_29, full_int_array_11, full_26)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_20 = split_10[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__8 = paddle._C_ops.add_(add__7, slice_20)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_21 = split_10[1]

        # builtin.combine: ([-1x192x28x28xf16, -1x32x28x28xf16]) <- (-1x192x28x28xf16, -1x32x28x28xf16)
        combine_16 = [concat_14, slice_21]

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x224x28x28xf16) <- ([-1x192x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_27)

        # builtin.combine: ([-1x512x28x28xf16, -1x224x28x28xf16]) <- (-1x512x28x28xf16, -1x224x28x28xf16)
        combine_17 = [add__8, concat_16]

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x736x28x28xf16) <- ([-1x512x28x28xf16, -1x224x28x28xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_28)

        # pd_op.batch_norm_: (-1x736x28x28xf16, 736xf32, 736xf32, 736xf32, 736xf32, None) <- (-1x736x28x28xf16, 736xf32, 736xf32, 736xf32, 736xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_17, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x736x28x28xf16) <- (-1x736x28x28xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x736x28x28xf16, 320x736x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__30, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__31, parameter_159, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__32 = paddle._C_ops.relu_(batch_norm__192)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__32, parameter_164, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_11 = paddle._C_ops.split(conv2d_32, full_int_array_12, full_29)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_22 = split_11[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__9 = paddle._C_ops.add_(add__8, slice_22)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_23 = split_11[1]

        # builtin.combine: ([-1x224x28x28xf16, -1x32x28x28xf16]) <- (-1x224x28x28xf16, -1x32x28x28xf16)
        combine_18 = [concat_16, slice_23]

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x28x28xf16) <- ([-1x224x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_30)

        # builtin.combine: ([-1x512x28x28xf16, -1x256x28x28xf16]) <- (-1x512x28x28xf16, -1x256x28x28xf16)
        combine_19 = [add__9, concat_18]

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x28x28xf16) <- ([-1x512x28x28xf16, -1x256x28x28xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_31)

        # pd_op.batch_norm_: (-1x768x28x28xf16, 768xf32, 768xf32, 768xf32, 768xf32, None) <- (-1x768x28x28xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_19, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x768x28x28xf16) <- (-1x768x28x28xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__198)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x768x28x28xf16, 320x768x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu__33, parameter_169, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__34, parameter_174, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__35 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__35, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(conv2d_35, full_int_array_13, full_32)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_24 = split_12[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__10 = paddle._C_ops.add_(add__9, slice_24)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_25 = split_12[1]

        # builtin.combine: ([-1x256x28x28xf16, -1x32x28x28xf16]) <- (-1x256x28x28xf16, -1x32x28x28xf16)
        combine_20 = [concat_18, slice_25]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x288x28x28xf16) <- ([-1x256x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_33)

        # builtin.combine: ([-1x512x28x28xf16, -1x288x28x28xf16]) <- (-1x512x28x28xf16, -1x288x28x28xf16)
        combine_21 = [add__10, concat_20]

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x800x28x28xf16) <- ([-1x512x28x28xf16, -1x288x28x28xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_34)

        # pd_op.batch_norm_: (-1x800x28x28xf16, 800xf32, 800xf32, 800xf32, 800xf32, None) <- (-1x800x28x28xf16, 800xf32, 800xf32, 800xf32, 800xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_21, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x800x28x28xf16) <- (-1x800x28x28xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x800x28x28xf16, 320x800x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu__36, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__37 = paddle._C_ops.relu_(batch_norm__222)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x320x28x28xf16, 320x8x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__37, parameter_189, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x544x28x28xf16) <- (-1x320x28x28xf16, 544x320x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__38, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [512, 32]

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x28x28xf16, -1x32x28x28xf16]) <- (-1x544x28x28xf16, 2xi64, 1xi32)
        split_13 = paddle._C_ops.split(conv2d_38, full_int_array_14, full_35)

        # builtin.slice: (-1x512x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_26 = split_13[0]

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__11 = paddle._C_ops.add_(add__10, slice_26)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x512x28x28xf16, -1x32x28x28xf16])
        slice_27 = split_13[1]

        # builtin.combine: ([-1x288x28x28xf16, -1x32x28x28xf16]) <- (-1x288x28x28xf16, -1x32x28x28xf16)
        combine_22 = [concat_20, slice_27]

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x28x28xf16) <- ([-1x288x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_36)

        # builtin.combine: ([-1x512x28x28xf16, -1x320x28x28xf16]) <- (-1x512x28x28xf16, -1x320x28x28xf16)
        combine_23 = [add__11, concat_22]

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x28x28xf16) <- ([-1x512x28x28xf16, -1x320x28x28xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_37)

        # pd_op.batch_norm_: (-1x832x28x28xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x28x28xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_23, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x28x28xf16) <- (-1x832x28x28xf16)
        relu__39 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.conv2d: (-1x1088x14x14xf16) <- (-1x832x28x28xf16, 1088x832x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__39, parameter_199, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [1024, 64]

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x64x14x14xf16]) <- (-1x1088x14x14xf16, 2xi64, 1xi32)
        split_14 = paddle._C_ops.split(conv2d_39, full_int_array_15, full_38)

        # pd_op.batch_norm_: (-1x832x28x28xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x28x28xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_23, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x28x28xf16) <- (-1x832x28x28xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x640x28x28xf16) <- (-1x832x28x28xf16, 640x832x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__40, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x28x28xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x28x28xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x28x28xf16) <- (-1x640x28x28xf16)
        relu__41 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x28x28xf16, 640x16x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__41, parameter_209, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__42 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu__42, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(conv2d_42, full_int_array_16, full_39)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x64x14x14xf16])
        slice_28 = split_14[0]

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_29 = split_15[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__12 = paddle._C_ops.add_(slice_28, slice_29)

        # builtin.slice: (-1x64x14x14xf16) <- ([-1x1024x14x14xf16, -1x64x14x14xf16])
        slice_30 = split_14[1]

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_31 = split_15[1]

        # builtin.combine: ([-1x64x14x14xf16, -1x32x14x14xf16]) <- (-1x64x14x14xf16, -1x32x14x14xf16)
        combine_24 = [slice_30, slice_31]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x14x14xf16) <- ([-1x64x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_40)

        # builtin.combine: ([-1x1024x14x14xf16, -1x96x14x14xf16]) <- (-1x1024x14x14xf16, -1x96x14x14xf16)
        combine_25 = [add__12, concat_24]

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1120x14x14xf16) <- ([-1x1024x14x14xf16, -1x96x14x14xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_41)

        # pd_op.batch_norm_: (-1x1120x14x14xf16, 1120xf32, 1120xf32, 1120xf32, 1120xf32, None) <- (-1x1120x14x14xf16, 1120xf32, 1120xf32, 1120xf32, 1120xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_25, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1120x14x14xf16) <- (-1x1120x14x14xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1120x14x14xf16, 640x1120x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__43, parameter_219, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_220, parameter_221, parameter_222, parameter_223, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu__44, parameter_224, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_225, parameter_226, parameter_227, parameter_228, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__45, parameter_229, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_16 = paddle._C_ops.split(conv2d_45, full_int_array_17, full_42)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_32 = split_16[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__13 = paddle._C_ops.add_(add__12, slice_32)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_33 = split_16[1]

        # builtin.combine: ([-1x96x14x14xf16, -1x32x14x14xf16]) <- (-1x96x14x14xf16, -1x32x14x14xf16)
        combine_26 = [concat_24, slice_33]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x14x14xf16) <- ([-1x96x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_43)

        # builtin.combine: ([-1x1024x14x14xf16, -1x128x14x14xf16]) <- (-1x1024x14x14xf16, -1x128x14x14xf16)
        combine_27 = [add__13, concat_26]

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1152x14x14xf16) <- ([-1x1024x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_44)

        # pd_op.batch_norm_: (-1x1152x14x14xf16, 1152xf32, 1152xf32, 1152xf32, 1152xf32, None) <- (-1x1152x14x14xf16, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_27, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1152x14x14xf16, 640x1152x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__46, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__47 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__47, parameter_239, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_240, parameter_241, parameter_242, parameter_243, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu__48, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_17 = paddle._C_ops.split(conv2d_48, full_int_array_18, full_45)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_34 = split_17[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__14 = paddle._C_ops.add_(add__13, slice_34)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_35 = split_17[1]

        # builtin.combine: ([-1x128x14x14xf16, -1x32x14x14xf16]) <- (-1x128x14x14xf16, -1x32x14x14xf16)
        combine_28 = [concat_26, slice_35]

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x14x14xf16) <- ([-1x128x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_28, full_46)

        # builtin.combine: ([-1x1024x14x14xf16, -1x160x14x14xf16]) <- (-1x1024x14x14xf16, -1x160x14x14xf16)
        combine_29 = [add__14, concat_28]

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1184x14x14xf16) <- ([-1x1024x14x14xf16, -1x160x14x14xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_29, full_47)

        # pd_op.batch_norm_: (-1x1184x14x14xf16, 1184xf32, 1184xf32, 1184xf32, 1184xf32, None) <- (-1x1184x14x14xf16, 1184xf32, 1184xf32, 1184xf32, 1184xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_29, parameter_245, parameter_246, parameter_247, parameter_248, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1184x14x14xf16) <- (-1x1184x14x14xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1184x14x14xf16, 640x1184x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__49, parameter_249, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_250, parameter_251, parameter_252, parameter_253, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__50 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(relu__50, parameter_254, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_255, parameter_256, parameter_257, parameter_258, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__51 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__51, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(conv2d_51, full_int_array_19, full_48)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_36 = split_18[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__15 = paddle._C_ops.add_(add__14, slice_36)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_37 = split_18[1]

        # builtin.combine: ([-1x160x14x14xf16, -1x32x14x14xf16]) <- (-1x160x14x14xf16, -1x32x14x14xf16)
        combine_30 = [concat_28, slice_37]

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x14x14xf16) <- ([-1x160x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_30, full_49)

        # builtin.combine: ([-1x1024x14x14xf16, -1x192x14x14xf16]) <- (-1x1024x14x14xf16, -1x192x14x14xf16)
        combine_31 = [add__15, concat_30]

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1216x14x14xf16) <- ([-1x1024x14x14xf16, -1x192x14x14xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_31, full_50)

        # pd_op.batch_norm_: (-1x1216x14x14xf16, 1216xf32, 1216xf32, 1216xf32, 1216xf32, None) <- (-1x1216x14x14xf16, 1216xf32, 1216xf32, 1216xf32, 1216xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_31, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1216x14x14xf16) <- (-1x1216x14x14xf16)
        relu__52 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1216x14x14xf16, 640x1216x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__52, parameter_264, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__53 = paddle._C_ops.relu_(batch_norm__318)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__53, parameter_269, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__54 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__54, parameter_274, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_19 = paddle._C_ops.split(conv2d_54, full_int_array_20, full_51)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_38 = split_19[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__16 = paddle._C_ops.add_(add__15, slice_38)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_39 = split_19[1]

        # builtin.combine: ([-1x192x14x14xf16, -1x32x14x14xf16]) <- (-1x192x14x14xf16, -1x32x14x14xf16)
        combine_32 = [concat_30, slice_39]

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x224x14x14xf16) <- ([-1x192x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_32, full_52)

        # builtin.combine: ([-1x1024x14x14xf16, -1x224x14x14xf16]) <- (-1x1024x14x14xf16, -1x224x14x14xf16)
        combine_33 = [add__16, concat_32]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1248x14x14xf16) <- ([-1x1024x14x14xf16, -1x224x14x14xf16], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_33, full_53)

        # pd_op.batch_norm_: (-1x1248x14x14xf16, 1248xf32, 1248xf32, 1248xf32, 1248xf32, None) <- (-1x1248x14x14xf16, 1248xf32, 1248xf32, 1248xf32, 1248xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_33, parameter_275, parameter_276, parameter_277, parameter_278, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1248x14x14xf16) <- (-1x1248x14x14xf16)
        relu__55 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1248x14x14xf16, 640x1248x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(relu__55, parameter_279, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_280, parameter_281, parameter_282, parameter_283, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__56, parameter_284, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_285, parameter_286, parameter_287, parameter_288, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__57 = paddle._C_ops.relu_(batch_norm__342)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu__57, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_20 = paddle._C_ops.split(conv2d_57, full_int_array_21, full_54)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_40 = split_20[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__17 = paddle._C_ops.add_(add__16, slice_40)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_41 = split_20[1]

        # builtin.combine: ([-1x224x14x14xf16, -1x32x14x14xf16]) <- (-1x224x14x14xf16, -1x32x14x14xf16)
        combine_34 = [concat_32, slice_41]

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x14x14xf16) <- ([-1x224x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_34, full_55)

        # builtin.combine: ([-1x1024x14x14xf16, -1x256x14x14xf16]) <- (-1x1024x14x14xf16, -1x256x14x14xf16)
        combine_35 = [add__17, concat_34]

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1280x14x14xf16) <- ([-1x1024x14x14xf16, -1x256x14x14xf16], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_35, full_56)

        # pd_op.batch_norm_: (-1x1280x14x14xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x14x14xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_35, parameter_290, parameter_291, parameter_292, parameter_293, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x14x14xf16) <- (-1x1280x14x14xf16)
        relu__58 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1280x14x14xf16, 640x1280x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(relu__58, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__59 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu__59, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__360)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(relu__60, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(conv2d_60, full_int_array_22, full_57)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_42 = split_21[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__18 = paddle._C_ops.add_(add__17, slice_42)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_43 = split_21[1]

        # builtin.combine: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x256x14x14xf16, -1x32x14x14xf16)
        combine_36 = [concat_34, slice_43]

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x288x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_36, full_58)

        # builtin.combine: ([-1x1024x14x14xf16, -1x288x14x14xf16]) <- (-1x1024x14x14xf16, -1x288x14x14xf16)
        combine_37 = [add__18, concat_36]

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1312x14x14xf16) <- ([-1x1024x14x14xf16, -1x288x14x14xf16], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_37, full_59)

        # pd_op.batch_norm_: (-1x1312x14x14xf16, 1312xf32, 1312xf32, 1312xf32, 1312xf32, None) <- (-1x1312x14x14xf16, 1312xf32, 1312xf32, 1312xf32, 1312xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_37, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1312x14x14xf16) <- (-1x1312x14x14xf16)
        relu__61 = paddle._C_ops.relu_(batch_norm__366)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1312x14x14xf16, 640x1312x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(relu__61, parameter_309, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__62 = paddle._C_ops.relu_(batch_norm__372)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu__62, parameter_314, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_315, parameter_316, parameter_317, parameter_318, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__63 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(relu__63, parameter_319, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_22 = paddle._C_ops.split(conv2d_63, full_int_array_23, full_60)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_44 = split_22[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__19 = paddle._C_ops.add_(add__18, slice_44)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_45 = split_22[1]

        # builtin.combine: ([-1x288x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, -1x32x14x14xf16)
        combine_38 = [concat_36, slice_45]

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x14x14xf16) <- ([-1x288x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_38, full_61)

        # builtin.combine: ([-1x1024x14x14xf16, -1x320x14x14xf16]) <- (-1x1024x14x14xf16, -1x320x14x14xf16)
        combine_39 = [add__19, concat_38]

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1344x14x14xf16) <- ([-1x1024x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_39, full_62)

        # pd_op.batch_norm_: (-1x1344x14x14xf16, 1344xf32, 1344xf32, 1344xf32, 1344xf32, None) <- (-1x1344x14x14xf16, 1344xf32, 1344xf32, 1344xf32, 1344xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_39, parameter_320, parameter_321, parameter_322, parameter_323, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1344x14x14xf16) <- (-1x1344x14x14xf16)
        relu__64 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1344x14x14xf16, 640x1344x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu__64, parameter_324, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_325, parameter_326, parameter_327, parameter_328, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__65 = paddle._C_ops.relu_(batch_norm__390)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_65 = paddle._C_ops.conv2d(relu__65, parameter_329, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_330, parameter_331, parameter_332, parameter_333, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__66 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu__66, parameter_334, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_23 = paddle._C_ops.split(conv2d_66, full_int_array_24, full_63)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_46 = split_23[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__20 = paddle._C_ops.add_(add__19, slice_46)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_47 = split_23[1]

        # builtin.combine: ([-1x320x14x14xf16, -1x32x14x14xf16]) <- (-1x320x14x14xf16, -1x32x14x14xf16)
        combine_40 = [concat_38, slice_47]

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x352x14x14xf16) <- ([-1x320x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_40, full_64)

        # builtin.combine: ([-1x1024x14x14xf16, -1x352x14x14xf16]) <- (-1x1024x14x14xf16, -1x352x14x14xf16)
        combine_41 = [add__20, concat_40]

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1376x14x14xf16) <- ([-1x1024x14x14xf16, -1x352x14x14xf16], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_41, full_65)

        # pd_op.batch_norm_: (-1x1376x14x14xf16, 1376xf32, 1376xf32, 1376xf32, 1376xf32, None) <- (-1x1376x14x14xf16, 1376xf32, 1376xf32, 1376xf32, 1376xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_41, parameter_335, parameter_336, parameter_337, parameter_338, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1376x14x14xf16) <- (-1x1376x14x14xf16)
        relu__67 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1376x14x14xf16, 640x1376x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__67, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_340, parameter_341, parameter_342, parameter_343, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__68 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(relu__68, parameter_344, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_345, parameter_346, parameter_347, parameter_348, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__69 = paddle._C_ops.relu_(batch_norm__414)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__69, parameter_349, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(conv2d_69, full_int_array_25, full_66)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_48 = split_24[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__21 = paddle._C_ops.add_(add__20, slice_48)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_49 = split_24[1]

        # builtin.combine: ([-1x352x14x14xf16, -1x32x14x14xf16]) <- (-1x352x14x14xf16, -1x32x14x14xf16)
        combine_42 = [concat_40, slice_49]

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x14x14xf16) <- ([-1x352x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_42, full_67)

        # builtin.combine: ([-1x1024x14x14xf16, -1x384x14x14xf16]) <- (-1x1024x14x14xf16, -1x384x14x14xf16)
        combine_43 = [add__21, concat_42]

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1408x14x14xf16) <- ([-1x1024x14x14xf16, -1x384x14x14xf16], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_43, full_68)

        # pd_op.batch_norm_: (-1x1408x14x14xf16, 1408xf32, 1408xf32, 1408xf32, 1408xf32, None) <- (-1x1408x14x14xf16, 1408xf32, 1408xf32, 1408xf32, 1408xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_43, parameter_350, parameter_351, parameter_352, parameter_353, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1408x14x14xf16) <- (-1x1408x14x14xf16)
        relu__70 = paddle._C_ops.relu_(batch_norm__420)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1408x14x14xf16, 640x1408x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(relu__70, parameter_354, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_355, parameter_356, parameter_357, parameter_358, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__71 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_71 = paddle._C_ops.conv2d(relu__71, parameter_359, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_360, parameter_361, parameter_362, parameter_363, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__72 = paddle._C_ops.relu_(batch_norm__432)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_72 = paddle._C_ops.conv2d(relu__72, parameter_364, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_25 = paddle._C_ops.split(conv2d_72, full_int_array_26, full_69)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_50 = split_25[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__22 = paddle._C_ops.add_(add__21, slice_50)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_51 = split_25[1]

        # builtin.combine: ([-1x384x14x14xf16, -1x32x14x14xf16]) <- (-1x384x14x14xf16, -1x32x14x14xf16)
        combine_44 = [concat_42, slice_51]

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x384x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_44 = paddle._C_ops.concat(combine_44, full_70)

        # builtin.combine: ([-1x1024x14x14xf16, -1x416x14x14xf16]) <- (-1x1024x14x14xf16, -1x416x14x14xf16)
        combine_45 = [add__22, concat_44]

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1440x14x14xf16) <- ([-1x1024x14x14xf16, -1x416x14x14xf16], 1xi32)
        concat_45 = paddle._C_ops.concat(combine_45, full_71)

        # pd_op.batch_norm_: (-1x1440x14x14xf16, 1440xf32, 1440xf32, 1440xf32, 1440xf32, None) <- (-1x1440x14x14xf16, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_45, parameter_365, parameter_366, parameter_367, parameter_368, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1440x14x14xf16) <- (-1x1440x14x14xf16)
        relu__73 = paddle._C_ops.relu_(batch_norm__438)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1440x14x14xf16, 640x1440x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(relu__73, parameter_369, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_370, parameter_371, parameter_372, parameter_373, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__74 = paddle._C_ops.relu_(batch_norm__444)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_74 = paddle._C_ops.conv2d(relu__74, parameter_374, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_375, parameter_376, parameter_377, parameter_378, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__75 = paddle._C_ops.relu_(batch_norm__450)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(relu__75, parameter_379, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_27 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_26 = paddle._C_ops.split(conv2d_75, full_int_array_27, full_72)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_52 = split_26[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__23 = paddle._C_ops.add_(add__22, slice_52)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_53 = split_26[1]

        # builtin.combine: ([-1x416x14x14xf16, -1x32x14x14xf16]) <- (-1x416x14x14xf16, -1x32x14x14xf16)
        combine_46 = [concat_44, slice_53]

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x448x14x14xf16) <- ([-1x416x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_46 = paddle._C_ops.concat(combine_46, full_73)

        # builtin.combine: ([-1x1024x14x14xf16, -1x448x14x14xf16]) <- (-1x1024x14x14xf16, -1x448x14x14xf16)
        combine_47 = [add__23, concat_46]

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1472x14x14xf16) <- ([-1x1024x14x14xf16, -1x448x14x14xf16], 1xi32)
        concat_47 = paddle._C_ops.concat(combine_47, full_74)

        # pd_op.batch_norm_: (-1x1472x14x14xf16, 1472xf32, 1472xf32, 1472xf32, 1472xf32, None) <- (-1x1472x14x14xf16, 1472xf32, 1472xf32, 1472xf32, 1472xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_47, parameter_380, parameter_381, parameter_382, parameter_383, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1472x14x14xf16) <- (-1x1472x14x14xf16)
        relu__76 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1472x14x14xf16, 640x1472x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu__76, parameter_384, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_385, parameter_386, parameter_387, parameter_388, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__77 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_77 = paddle._C_ops.conv2d(relu__77, parameter_389, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_390, parameter_391, parameter_392, parameter_393, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__78 = paddle._C_ops.relu_(batch_norm__468)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_78 = paddle._C_ops.conv2d(relu__78, parameter_394, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(conv2d_78, full_int_array_28, full_75)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_54 = split_27[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__24 = paddle._C_ops.add_(add__23, slice_54)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_55 = split_27[1]

        # builtin.combine: ([-1x448x14x14xf16, -1x32x14x14xf16]) <- (-1x448x14x14xf16, -1x32x14x14xf16)
        combine_48 = [concat_46, slice_55]

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x480x14x14xf16) <- ([-1x448x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_48 = paddle._C_ops.concat(combine_48, full_76)

        # builtin.combine: ([-1x1024x14x14xf16, -1x480x14x14xf16]) <- (-1x1024x14x14xf16, -1x480x14x14xf16)
        combine_49 = [add__24, concat_48]

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1504x14x14xf16) <- ([-1x1024x14x14xf16, -1x480x14x14xf16], 1xi32)
        concat_49 = paddle._C_ops.concat(combine_49, full_77)

        # pd_op.batch_norm_: (-1x1504x14x14xf16, 1504xf32, 1504xf32, 1504xf32, 1504xf32, None) <- (-1x1504x14x14xf16, 1504xf32, 1504xf32, 1504xf32, 1504xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_49, parameter_395, parameter_396, parameter_397, parameter_398, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1504x14x14xf16) <- (-1x1504x14x14xf16)
        relu__79 = paddle._C_ops.relu_(batch_norm__474)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1504x14x14xf16, 640x1504x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(relu__79, parameter_399, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_400, parameter_401, parameter_402, parameter_403, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__80 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(relu__80, parameter_404, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_405, parameter_406, parameter_407, parameter_408, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__81 = paddle._C_ops.relu_(batch_norm__486)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(relu__81, parameter_409, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_29 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_28 = paddle._C_ops.split(conv2d_81, full_int_array_29, full_78)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_56 = split_28[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__25 = paddle._C_ops.add_(add__24, slice_56)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_57 = split_28[1]

        # builtin.combine: ([-1x480x14x14xf16, -1x32x14x14xf16]) <- (-1x480x14x14xf16, -1x32x14x14xf16)
        combine_50 = [concat_48, slice_57]

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x480x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_50 = paddle._C_ops.concat(combine_50, full_79)

        # builtin.combine: ([-1x1024x14x14xf16, -1x512x14x14xf16]) <- (-1x1024x14x14xf16, -1x512x14x14xf16)
        combine_51 = [add__25, concat_50]

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1536x14x14xf16) <- ([-1x1024x14x14xf16, -1x512x14x14xf16], 1xi32)
        concat_51 = paddle._C_ops.concat(combine_51, full_80)

        # pd_op.batch_norm_: (-1x1536x14x14xf16, 1536xf32, 1536xf32, 1536xf32, 1536xf32, None) <- (-1x1536x14x14xf16, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_51, parameter_410, parameter_411, parameter_412, parameter_413, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1536x14x14xf16) <- (-1x1536x14x14xf16)
        relu__82 = paddle._C_ops.relu_(batch_norm__492)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1536x14x14xf16, 640x1536x1x1xf16)
        conv2d_82 = paddle._C_ops.conv2d(relu__82, parameter_414, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_415, parameter_416, parameter_417, parameter_418, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__83 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_83 = paddle._C_ops.conv2d(relu__83, parameter_419, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_420, parameter_421, parameter_422, parameter_423, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__84 = paddle._C_ops.relu_(batch_norm__504)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_84 = paddle._C_ops.conv2d(relu__84, parameter_424, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_30 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_29 = paddle._C_ops.split(conv2d_84, full_int_array_30, full_81)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_58 = split_29[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__26 = paddle._C_ops.add_(add__25, slice_58)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_59 = split_29[1]

        # builtin.combine: ([-1x512x14x14xf16, -1x32x14x14xf16]) <- (-1x512x14x14xf16, -1x32x14x14xf16)
        combine_52 = [concat_50, slice_59]

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x544x14x14xf16) <- ([-1x512x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_52 = paddle._C_ops.concat(combine_52, full_82)

        # builtin.combine: ([-1x1024x14x14xf16, -1x544x14x14xf16]) <- (-1x1024x14x14xf16, -1x544x14x14xf16)
        combine_53 = [add__26, concat_52]

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1568x14x14xf16) <- ([-1x1024x14x14xf16, -1x544x14x14xf16], 1xi32)
        concat_53 = paddle._C_ops.concat(combine_53, full_83)

        # pd_op.batch_norm_: (-1x1568x14x14xf16, 1568xf32, 1568xf32, 1568xf32, 1568xf32, None) <- (-1x1568x14x14xf16, 1568xf32, 1568xf32, 1568xf32, 1568xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_53, parameter_425, parameter_426, parameter_427, parameter_428, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1568x14x14xf16) <- (-1x1568x14x14xf16)
        relu__85 = paddle._C_ops.relu_(batch_norm__510)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1568x14x14xf16, 640x1568x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu__85, parameter_429, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_430, parameter_431, parameter_432, parameter_433, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__86 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_86 = paddle._C_ops.conv2d(relu__86, parameter_434, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_435, parameter_436, parameter_437, parameter_438, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__87 = paddle._C_ops.relu_(batch_norm__522)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_87 = paddle._C_ops.conv2d(relu__87, parameter_439, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(conv2d_87, full_int_array_31, full_84)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_60 = split_30[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__27 = paddle._C_ops.add_(add__26, slice_60)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_61 = split_30[1]

        # builtin.combine: ([-1x544x14x14xf16, -1x32x14x14xf16]) <- (-1x544x14x14xf16, -1x32x14x14xf16)
        combine_54 = [concat_52, slice_61]

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x576x14x14xf16) <- ([-1x544x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_54 = paddle._C_ops.concat(combine_54, full_85)

        # builtin.combine: ([-1x1024x14x14xf16, -1x576x14x14xf16]) <- (-1x1024x14x14xf16, -1x576x14x14xf16)
        combine_55 = [add__27, concat_54]

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1600x14x14xf16) <- ([-1x1024x14x14xf16, -1x576x14x14xf16], 1xi32)
        concat_55 = paddle._C_ops.concat(combine_55, full_86)

        # pd_op.batch_norm_: (-1x1600x14x14xf16, 1600xf32, 1600xf32, 1600xf32, 1600xf32, None) <- (-1x1600x14x14xf16, 1600xf32, 1600xf32, 1600xf32, 1600xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_55, parameter_440, parameter_441, parameter_442, parameter_443, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1600x14x14xf16) <- (-1x1600x14x14xf16)
        relu__88 = paddle._C_ops.relu_(batch_norm__528)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1600x14x14xf16, 640x1600x1x1xf16)
        conv2d_88 = paddle._C_ops.conv2d(relu__88, parameter_444, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_445, parameter_446, parameter_447, parameter_448, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__89 = paddle._C_ops.relu_(batch_norm__534)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_89 = paddle._C_ops.conv2d(relu__89, parameter_449, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_450, parameter_451, parameter_452, parameter_453, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__90 = paddle._C_ops.relu_(batch_norm__540)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_90 = paddle._C_ops.conv2d(relu__90, parameter_454, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_31 = paddle._C_ops.split(conv2d_90, full_int_array_32, full_87)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_62 = split_31[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__28 = paddle._C_ops.add_(add__27, slice_62)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_63 = split_31[1]

        # builtin.combine: ([-1x576x14x14xf16, -1x32x14x14xf16]) <- (-1x576x14x14xf16, -1x32x14x14xf16)
        combine_56 = [concat_54, slice_63]

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x608x14x14xf16) <- ([-1x576x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_56 = paddle._C_ops.concat(combine_56, full_88)

        # builtin.combine: ([-1x1024x14x14xf16, -1x608x14x14xf16]) <- (-1x1024x14x14xf16, -1x608x14x14xf16)
        combine_57 = [add__28, concat_56]

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1632x14x14xf16) <- ([-1x1024x14x14xf16, -1x608x14x14xf16], 1xi32)
        concat_57 = paddle._C_ops.concat(combine_57, full_89)

        # pd_op.batch_norm_: (-1x1632x14x14xf16, 1632xf32, 1632xf32, 1632xf32, 1632xf32, None) <- (-1x1632x14x14xf16, 1632xf32, 1632xf32, 1632xf32, 1632xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_57, parameter_455, parameter_456, parameter_457, parameter_458, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1632x14x14xf16) <- (-1x1632x14x14xf16)
        relu__91 = paddle._C_ops.relu_(batch_norm__546)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1632x14x14xf16, 640x1632x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(relu__91, parameter_459, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_460, parameter_461, parameter_462, parameter_463, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__92 = paddle._C_ops.relu_(batch_norm__552)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(relu__92, parameter_464, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_465, parameter_466, parameter_467, parameter_468, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__93 = paddle._C_ops.relu_(batch_norm__558)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_93 = paddle._C_ops.conv2d(relu__93, parameter_469, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_33 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_32 = paddle._C_ops.split(conv2d_93, full_int_array_33, full_90)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_64 = split_32[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__29 = paddle._C_ops.add_(add__28, slice_64)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_65 = split_32[1]

        # builtin.combine: ([-1x608x14x14xf16, -1x32x14x14xf16]) <- (-1x608x14x14xf16, -1x32x14x14xf16)
        combine_58 = [concat_56, slice_65]

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x640x14x14xf16) <- ([-1x608x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_58 = paddle._C_ops.concat(combine_58, full_91)

        # builtin.combine: ([-1x1024x14x14xf16, -1x640x14x14xf16]) <- (-1x1024x14x14xf16, -1x640x14x14xf16)
        combine_59 = [add__29, concat_58]

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1664x14x14xf16) <- ([-1x1024x14x14xf16, -1x640x14x14xf16], 1xi32)
        concat_59 = paddle._C_ops.concat(combine_59, full_92)

        # pd_op.batch_norm_: (-1x1664x14x14xf16, 1664xf32, 1664xf32, 1664xf32, 1664xf32, None) <- (-1x1664x14x14xf16, 1664xf32, 1664xf32, 1664xf32, 1664xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_59, parameter_470, parameter_471, parameter_472, parameter_473, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1664x14x14xf16) <- (-1x1664x14x14xf16)
        relu__94 = paddle._C_ops.relu_(batch_norm__564)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1664x14x14xf16, 640x1664x1x1xf16)
        conv2d_94 = paddle._C_ops.conv2d(relu__94, parameter_474, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_475, parameter_476, parameter_477, parameter_478, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__95 = paddle._C_ops.relu_(batch_norm__570)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_95 = paddle._C_ops.conv2d(relu__95, parameter_479, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_480, parameter_481, parameter_482, parameter_483, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__96 = paddle._C_ops.relu_(batch_norm__576)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(relu__96, parameter_484, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_34 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(conv2d_96, full_int_array_34, full_93)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_66 = split_33[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__30 = paddle._C_ops.add_(add__29, slice_66)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_67 = split_33[1]

        # builtin.combine: ([-1x640x14x14xf16, -1x32x14x14xf16]) <- (-1x640x14x14xf16, -1x32x14x14xf16)
        combine_60 = [concat_58, slice_67]

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x672x14x14xf16) <- ([-1x640x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_60 = paddle._C_ops.concat(combine_60, full_94)

        # builtin.combine: ([-1x1024x14x14xf16, -1x672x14x14xf16]) <- (-1x1024x14x14xf16, -1x672x14x14xf16)
        combine_61 = [add__30, concat_60]

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1696x14x14xf16) <- ([-1x1024x14x14xf16, -1x672x14x14xf16], 1xi32)
        concat_61 = paddle._C_ops.concat(combine_61, full_95)

        # pd_op.batch_norm_: (-1x1696x14x14xf16, 1696xf32, 1696xf32, 1696xf32, 1696xf32, None) <- (-1x1696x14x14xf16, 1696xf32, 1696xf32, 1696xf32, 1696xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_61, parameter_485, parameter_486, parameter_487, parameter_488, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1696x14x14xf16) <- (-1x1696x14x14xf16)
        relu__97 = paddle._C_ops.relu_(batch_norm__582)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1696x14x14xf16, 640x1696x1x1xf16)
        conv2d_97 = paddle._C_ops.conv2d(relu__97, parameter_489, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_490, parameter_491, parameter_492, parameter_493, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__98 = paddle._C_ops.relu_(batch_norm__588)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_98 = paddle._C_ops.conv2d(relu__98, parameter_494, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_495, parameter_496, parameter_497, parameter_498, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__99 = paddle._C_ops.relu_(batch_norm__594)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_99 = paddle._C_ops.conv2d(relu__99, parameter_499, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_35 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_34 = paddle._C_ops.split(conv2d_99, full_int_array_35, full_96)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_68 = split_34[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__31 = paddle._C_ops.add_(add__30, slice_68)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_69 = split_34[1]

        # builtin.combine: ([-1x672x14x14xf16, -1x32x14x14xf16]) <- (-1x672x14x14xf16, -1x32x14x14xf16)
        combine_62 = [concat_60, slice_69]

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x704x14x14xf16) <- ([-1x672x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_62 = paddle._C_ops.concat(combine_62, full_97)

        # builtin.combine: ([-1x1024x14x14xf16, -1x704x14x14xf16]) <- (-1x1024x14x14xf16, -1x704x14x14xf16)
        combine_63 = [add__31, concat_62]

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1728x14x14xf16) <- ([-1x1024x14x14xf16, -1x704x14x14xf16], 1xi32)
        concat_63 = paddle._C_ops.concat(combine_63, full_98)

        # pd_op.batch_norm_: (-1x1728x14x14xf16, 1728xf32, 1728xf32, 1728xf32, 1728xf32, None) <- (-1x1728x14x14xf16, 1728xf32, 1728xf32, 1728xf32, 1728xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_63, parameter_500, parameter_501, parameter_502, parameter_503, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1728x14x14xf16) <- (-1x1728x14x14xf16)
        relu__100 = paddle._C_ops.relu_(batch_norm__600)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1728x14x14xf16, 640x1728x1x1xf16)
        conv2d_100 = paddle._C_ops.conv2d(relu__100, parameter_504, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_505, parameter_506, parameter_507, parameter_508, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__101 = paddle._C_ops.relu_(batch_norm__606)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_101 = paddle._C_ops.conv2d(relu__101, parameter_509, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_510, parameter_511, parameter_512, parameter_513, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__102 = paddle._C_ops.relu_(batch_norm__612)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_102 = paddle._C_ops.conv2d(relu__102, parameter_514, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_35 = paddle._C_ops.split(conv2d_102, full_int_array_36, full_99)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_70 = split_35[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__32 = paddle._C_ops.add_(add__31, slice_70)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_71 = split_35[1]

        # builtin.combine: ([-1x704x14x14xf16, -1x32x14x14xf16]) <- (-1x704x14x14xf16, -1x32x14x14xf16)
        combine_64 = [concat_62, slice_71]

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x736x14x14xf16) <- ([-1x704x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_64 = paddle._C_ops.concat(combine_64, full_100)

        # builtin.combine: ([-1x1024x14x14xf16, -1x736x14x14xf16]) <- (-1x1024x14x14xf16, -1x736x14x14xf16)
        combine_65 = [add__32, concat_64]

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1760x14x14xf16) <- ([-1x1024x14x14xf16, -1x736x14x14xf16], 1xi32)
        concat_65 = paddle._C_ops.concat(combine_65, full_101)

        # pd_op.batch_norm_: (-1x1760x14x14xf16, 1760xf32, 1760xf32, 1760xf32, 1760xf32, None) <- (-1x1760x14x14xf16, 1760xf32, 1760xf32, 1760xf32, 1760xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_65, parameter_515, parameter_516, parameter_517, parameter_518, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1760x14x14xf16) <- (-1x1760x14x14xf16)
        relu__103 = paddle._C_ops.relu_(batch_norm__618)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1760x14x14xf16, 640x1760x1x1xf16)
        conv2d_103 = paddle._C_ops.conv2d(relu__103, parameter_519, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_520, parameter_521, parameter_522, parameter_523, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__104 = paddle._C_ops.relu_(batch_norm__624)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_104 = paddle._C_ops.conv2d(relu__104, parameter_524, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_104, parameter_525, parameter_526, parameter_527, parameter_528, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__105 = paddle._C_ops.relu_(batch_norm__630)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_105 = paddle._C_ops.conv2d(relu__105, parameter_529, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(conv2d_105, full_int_array_37, full_102)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_72 = split_36[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__33 = paddle._C_ops.add_(add__32, slice_72)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_73 = split_36[1]

        # builtin.combine: ([-1x736x14x14xf16, -1x32x14x14xf16]) <- (-1x736x14x14xf16, -1x32x14x14xf16)
        combine_66 = [concat_64, slice_73]

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x14x14xf16) <- ([-1x736x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_66 = paddle._C_ops.concat(combine_66, full_103)

        # builtin.combine: ([-1x1024x14x14xf16, -1x768x14x14xf16]) <- (-1x1024x14x14xf16, -1x768x14x14xf16)
        combine_67 = [add__33, concat_66]

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1792x14x14xf16) <- ([-1x1024x14x14xf16, -1x768x14x14xf16], 1xi32)
        concat_67 = paddle._C_ops.concat(combine_67, full_104)

        # pd_op.batch_norm_: (-1x1792x14x14xf16, 1792xf32, 1792xf32, 1792xf32, 1792xf32, None) <- (-1x1792x14x14xf16, 1792xf32, 1792xf32, 1792xf32, 1792xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_67, parameter_530, parameter_531, parameter_532, parameter_533, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1792x14x14xf16) <- (-1x1792x14x14xf16)
        relu__106 = paddle._C_ops.relu_(batch_norm__636)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1792x14x14xf16, 640x1792x1x1xf16)
        conv2d_106 = paddle._C_ops.conv2d(relu__106, parameter_534, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_535, parameter_536, parameter_537, parameter_538, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__107 = paddle._C_ops.relu_(batch_norm__642)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_107 = paddle._C_ops.conv2d(relu__107, parameter_539, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_540, parameter_541, parameter_542, parameter_543, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__108 = paddle._C_ops.relu_(batch_norm__648)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_108 = paddle._C_ops.conv2d(relu__108, parameter_544, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_37 = paddle._C_ops.split(conv2d_108, full_int_array_38, full_105)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_74 = split_37[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__34 = paddle._C_ops.add_(add__33, slice_74)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_75 = split_37[1]

        # builtin.combine: ([-1x768x14x14xf16, -1x32x14x14xf16]) <- (-1x768x14x14xf16, -1x32x14x14xf16)
        combine_68 = [concat_66, slice_75]

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x800x14x14xf16) <- ([-1x768x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_68 = paddle._C_ops.concat(combine_68, full_106)

        # builtin.combine: ([-1x1024x14x14xf16, -1x800x14x14xf16]) <- (-1x1024x14x14xf16, -1x800x14x14xf16)
        combine_69 = [add__34, concat_68]

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1824x14x14xf16) <- ([-1x1024x14x14xf16, -1x800x14x14xf16], 1xi32)
        concat_69 = paddle._C_ops.concat(combine_69, full_107)

        # pd_op.batch_norm_: (-1x1824x14x14xf16, 1824xf32, 1824xf32, 1824xf32, 1824xf32, None) <- (-1x1824x14x14xf16, 1824xf32, 1824xf32, 1824xf32, 1824xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_69, parameter_545, parameter_546, parameter_547, parameter_548, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1824x14x14xf16) <- (-1x1824x14x14xf16)
        relu__109 = paddle._C_ops.relu_(batch_norm__654)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1824x14x14xf16, 640x1824x1x1xf16)
        conv2d_109 = paddle._C_ops.conv2d(relu__109, parameter_549, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_109, parameter_550, parameter_551, parameter_552, parameter_553, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__110 = paddle._C_ops.relu_(batch_norm__660)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_110 = paddle._C_ops.conv2d(relu__110, parameter_554, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_555, parameter_556, parameter_557, parameter_558, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__111 = paddle._C_ops.relu_(batch_norm__666)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_111 = paddle._C_ops.conv2d(relu__111, parameter_559, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_38 = paddle._C_ops.split(conv2d_111, full_int_array_39, full_108)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_76 = split_38[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__35 = paddle._C_ops.add_(add__34, slice_76)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_77 = split_38[1]

        # builtin.combine: ([-1x800x14x14xf16, -1x32x14x14xf16]) <- (-1x800x14x14xf16, -1x32x14x14xf16)
        combine_70 = [concat_68, slice_77]

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x14x14xf16) <- ([-1x800x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_70 = paddle._C_ops.concat(combine_70, full_109)

        # builtin.combine: ([-1x1024x14x14xf16, -1x832x14x14xf16]) <- (-1x1024x14x14xf16, -1x832x14x14xf16)
        combine_71 = [add__35, concat_70]

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1856x14x14xf16) <- ([-1x1024x14x14xf16, -1x832x14x14xf16], 1xi32)
        concat_71 = paddle._C_ops.concat(combine_71, full_110)

        # pd_op.batch_norm_: (-1x1856x14x14xf16, 1856xf32, 1856xf32, 1856xf32, 1856xf32, None) <- (-1x1856x14x14xf16, 1856xf32, 1856xf32, 1856xf32, 1856xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_71, parameter_560, parameter_561, parameter_562, parameter_563, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1856x14x14xf16) <- (-1x1856x14x14xf16)
        relu__112 = paddle._C_ops.relu_(batch_norm__672)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1856x14x14xf16, 640x1856x1x1xf16)
        conv2d_112 = paddle._C_ops.conv2d(relu__112, parameter_564, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_565, parameter_566, parameter_567, parameter_568, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__113 = paddle._C_ops.relu_(batch_norm__678)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_113 = paddle._C_ops.conv2d(relu__113, parameter_569, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_113, parameter_570, parameter_571, parameter_572, parameter_573, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__114 = paddle._C_ops.relu_(batch_norm__684)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_114 = paddle._C_ops.conv2d(relu__114, parameter_574, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_39 = paddle._C_ops.split(conv2d_114, full_int_array_40, full_111)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_78 = split_39[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__36 = paddle._C_ops.add_(add__35, slice_78)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_79 = split_39[1]

        # builtin.combine: ([-1x832x14x14xf16, -1x32x14x14xf16]) <- (-1x832x14x14xf16, -1x32x14x14xf16)
        combine_72 = [concat_70, slice_79]

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x864x14x14xf16) <- ([-1x832x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_72 = paddle._C_ops.concat(combine_72, full_112)

        # builtin.combine: ([-1x1024x14x14xf16, -1x864x14x14xf16]) <- (-1x1024x14x14xf16, -1x864x14x14xf16)
        combine_73 = [add__36, concat_72]

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1888x14x14xf16) <- ([-1x1024x14x14xf16, -1x864x14x14xf16], 1xi32)
        concat_73 = paddle._C_ops.concat(combine_73, full_113)

        # pd_op.batch_norm_: (-1x1888x14x14xf16, 1888xf32, 1888xf32, 1888xf32, 1888xf32, None) <- (-1x1888x14x14xf16, 1888xf32, 1888xf32, 1888xf32, 1888xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_73, parameter_575, parameter_576, parameter_577, parameter_578, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1888x14x14xf16) <- (-1x1888x14x14xf16)
        relu__115 = paddle._C_ops.relu_(batch_norm__690)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1888x14x14xf16, 640x1888x1x1xf16)
        conv2d_115 = paddle._C_ops.conv2d(relu__115, parameter_579, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_580, parameter_581, parameter_582, parameter_583, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__116 = paddle._C_ops.relu_(batch_norm__696)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_116 = paddle._C_ops.conv2d(relu__116, parameter_584, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_585, parameter_586, parameter_587, parameter_588, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__117 = paddle._C_ops.relu_(batch_norm__702)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_117 = paddle._C_ops.conv2d(relu__117, parameter_589, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_40 = paddle._C_ops.split(conv2d_117, full_int_array_41, full_114)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_80 = split_40[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__37 = paddle._C_ops.add_(add__36, slice_80)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_81 = split_40[1]

        # builtin.combine: ([-1x864x14x14xf16, -1x32x14x14xf16]) <- (-1x864x14x14xf16, -1x32x14x14xf16)
        combine_74 = [concat_72, slice_81]

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x896x14x14xf16) <- ([-1x864x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_74 = paddle._C_ops.concat(combine_74, full_115)

        # builtin.combine: ([-1x1024x14x14xf16, -1x896x14x14xf16]) <- (-1x1024x14x14xf16, -1x896x14x14xf16)
        combine_75 = [add__37, concat_74]

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1920x14x14xf16) <- ([-1x1024x14x14xf16, -1x896x14x14xf16], 1xi32)
        concat_75 = paddle._C_ops.concat(combine_75, full_116)

        # pd_op.batch_norm_: (-1x1920x14x14xf16, 1920xf32, 1920xf32, 1920xf32, 1920xf32, None) <- (-1x1920x14x14xf16, 1920xf32, 1920xf32, 1920xf32, 1920xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_75, parameter_590, parameter_591, parameter_592, parameter_593, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1920x14x14xf16) <- (-1x1920x14x14xf16)
        relu__118 = paddle._C_ops.relu_(batch_norm__708)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1920x14x14xf16, 640x1920x1x1xf16)
        conv2d_118 = paddle._C_ops.conv2d(relu__118, parameter_594, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_595, parameter_596, parameter_597, parameter_598, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__119 = paddle._C_ops.relu_(batch_norm__714)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_119 = paddle._C_ops.conv2d(relu__119, parameter_599, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_600, parameter_601, parameter_602, parameter_603, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__120 = paddle._C_ops.relu_(batch_norm__720)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_120 = paddle._C_ops.conv2d(relu__120, parameter_604, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_41 = paddle._C_ops.split(conv2d_120, full_int_array_42, full_117)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_82 = split_41[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__38 = paddle._C_ops.add_(add__37, slice_82)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_83 = split_41[1]

        # builtin.combine: ([-1x896x14x14xf16, -1x32x14x14xf16]) <- (-1x896x14x14xf16, -1x32x14x14xf16)
        combine_76 = [concat_74, slice_83]

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x928x14x14xf16) <- ([-1x896x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_76 = paddle._C_ops.concat(combine_76, full_118)

        # builtin.combine: ([-1x1024x14x14xf16, -1x928x14x14xf16]) <- (-1x1024x14x14xf16, -1x928x14x14xf16)
        combine_77 = [add__38, concat_76]

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1952x14x14xf16) <- ([-1x1024x14x14xf16, -1x928x14x14xf16], 1xi32)
        concat_77 = paddle._C_ops.concat(combine_77, full_119)

        # pd_op.batch_norm_: (-1x1952x14x14xf16, 1952xf32, 1952xf32, 1952xf32, 1952xf32, None) <- (-1x1952x14x14xf16, 1952xf32, 1952xf32, 1952xf32, 1952xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_77, parameter_605, parameter_606, parameter_607, parameter_608, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1952x14x14xf16) <- (-1x1952x14x14xf16)
        relu__121 = paddle._C_ops.relu_(batch_norm__726)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x1952x14x14xf16, 640x1952x1x1xf16)
        conv2d_121 = paddle._C_ops.conv2d(relu__121, parameter_609, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_121, parameter_610, parameter_611, parameter_612, parameter_613, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__122 = paddle._C_ops.relu_(batch_norm__732)

        # pd_op.conv2d: (-1x640x14x14xf16) <- (-1x640x14x14xf16, 640x16x3x3xf16)
        conv2d_122 = paddle._C_ops.conv2d(relu__122, parameter_614, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_615, parameter_616, parameter_617, parameter_618, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__123 = paddle._C_ops.relu_(batch_norm__738)

        # pd_op.conv2d: (-1x1056x14x14xf16) <- (-1x640x14x14xf16, 1056x640x1x1xf16)
        conv2d_123 = paddle._C_ops.conv2d(relu__123, parameter_619, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [1024, 32]

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1024x14x14xf16, -1x32x14x14xf16]) <- (-1x1056x14x14xf16, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(conv2d_123, full_int_array_43, full_120)

        # builtin.slice: (-1x1024x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_84 = split_42[0]

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__39 = paddle._C_ops.add_(add__38, slice_84)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x1024x14x14xf16, -1x32x14x14xf16])
        slice_85 = split_42[1]

        # builtin.combine: ([-1x928x14x14xf16, -1x32x14x14xf16]) <- (-1x928x14x14xf16, -1x32x14x14xf16)
        combine_78 = [concat_76, slice_85]

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x960x14x14xf16) <- ([-1x928x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_78 = paddle._C_ops.concat(combine_78, full_121)

        # builtin.combine: ([-1x1024x14x14xf16, -1x960x14x14xf16]) <- (-1x1024x14x14xf16, -1x960x14x14xf16)
        combine_79 = [add__39, concat_78]

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1984x14x14xf16) <- ([-1x1024x14x14xf16, -1x960x14x14xf16], 1xi32)
        concat_79 = paddle._C_ops.concat(combine_79, full_122)

        # pd_op.batch_norm_: (-1x1984x14x14xf16, 1984xf32, 1984xf32, 1984xf32, 1984xf32, None) <- (-1x1984x14x14xf16, 1984xf32, 1984xf32, 1984xf32, 1984xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_79, parameter_620, parameter_621, parameter_622, parameter_623, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1984x14x14xf16) <- (-1x1984x14x14xf16)
        relu__124 = paddle._C_ops.relu_(batch_norm__744)

        # pd_op.conv2d: (-1x2304x7x7xf16) <- (-1x1984x14x14xf16, 2304x1984x1x1xf16)
        conv2d_124 = paddle._C_ops.conv2d(relu__124, parameter_624, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [2048, 256]

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x2048x7x7xf16, -1x256x7x7xf16]) <- (-1x2304x7x7xf16, 2xi64, 1xi32)
        split_43 = paddle._C_ops.split(conv2d_124, full_int_array_44, full_123)

        # pd_op.batch_norm_: (-1x1984x14x14xf16, 1984xf32, 1984xf32, 1984xf32, 1984xf32, None) <- (-1x1984x14x14xf16, 1984xf32, 1984xf32, 1984xf32, 1984xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_79, parameter_625, parameter_626, parameter_627, parameter_628, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1984x14x14xf16) <- (-1x1984x14x14xf16)
        relu__125 = paddle._C_ops.relu_(batch_norm__750)

        # pd_op.conv2d: (-1x1280x14x14xf16) <- (-1x1984x14x14xf16, 1280x1984x1x1xf16)
        conv2d_125 = paddle._C_ops.conv2d(relu__125, parameter_629, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x14x14xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x14x14xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_125, parameter_630, parameter_631, parameter_632, parameter_633, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x14x14xf16) <- (-1x1280x14x14xf16)
        relu__126 = paddle._C_ops.relu_(batch_norm__756)

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x1280x14x14xf16, 1280x32x3x3xf16)
        conv2d_126 = paddle._C_ops.conv2d(relu__126, parameter_634, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_635, parameter_636, parameter_637, parameter_638, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu__127 = paddle._C_ops.relu_(batch_norm__762)

        # pd_op.conv2d: (-1x2176x7x7xf16) <- (-1x1280x7x7xf16, 2176x1280x1x1xf16)
        conv2d_127 = paddle._C_ops.conv2d(relu__127, parameter_639, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_45 = [2048, 128]

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x2048x7x7xf16, -1x128x7x7xf16]) <- (-1x2176x7x7xf16, 2xi64, 1xi32)
        split_44 = paddle._C_ops.split(conv2d_127, full_int_array_45, full_124)

        # builtin.slice: (-1x2048x7x7xf16) <- ([-1x2048x7x7xf16, -1x256x7x7xf16])
        slice_86 = split_43[0]

        # builtin.slice: (-1x2048x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_87 = split_44[0]

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__40 = paddle._C_ops.add_(slice_86, slice_87)

        # builtin.slice: (-1x256x7x7xf16) <- ([-1x2048x7x7xf16, -1x256x7x7xf16])
        slice_88 = split_43[1]

        # builtin.slice: (-1x128x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_89 = split_44[1]

        # builtin.combine: ([-1x256x7x7xf16, -1x128x7x7xf16]) <- (-1x256x7x7xf16, -1x128x7x7xf16)
        combine_80 = [slice_88, slice_89]

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x7x7xf16) <- ([-1x256x7x7xf16, -1x128x7x7xf16], 1xi32)
        concat_80 = paddle._C_ops.concat(combine_80, full_125)

        # builtin.combine: ([-1x2048x7x7xf16, -1x384x7x7xf16]) <- (-1x2048x7x7xf16, -1x384x7x7xf16)
        combine_81 = [add__40, concat_80]

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2432x7x7xf16) <- ([-1x2048x7x7xf16, -1x384x7x7xf16], 1xi32)
        concat_81 = paddle._C_ops.concat(combine_81, full_126)

        # pd_op.batch_norm_: (-1x2432x7x7xf16, 2432xf32, 2432xf32, 2432xf32, 2432xf32, None) <- (-1x2432x7x7xf16, 2432xf32, 2432xf32, 2432xf32, 2432xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_81, parameter_640, parameter_641, parameter_642, parameter_643, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x2432x7x7xf16) <- (-1x2432x7x7xf16)
        relu__128 = paddle._C_ops.relu_(batch_norm__768)

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x2432x7x7xf16, 1280x2432x1x1xf16)
        conv2d_128 = paddle._C_ops.conv2d(relu__128, parameter_644, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_645, parameter_646, parameter_647, parameter_648, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu__129 = paddle._C_ops.relu_(batch_norm__774)

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16, 1280x32x3x3xf16)
        conv2d_129 = paddle._C_ops.conv2d(relu__129, parameter_649, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_129, parameter_650, parameter_651, parameter_652, parameter_653, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu__130 = paddle._C_ops.relu_(batch_norm__780)

        # pd_op.conv2d: (-1x2176x7x7xf16) <- (-1x1280x7x7xf16, 2176x1280x1x1xf16)
        conv2d_130 = paddle._C_ops.conv2d(relu__130, parameter_654, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_46 = [2048, 128]

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x2048x7x7xf16, -1x128x7x7xf16]) <- (-1x2176x7x7xf16, 2xi64, 1xi32)
        split_45 = paddle._C_ops.split(conv2d_130, full_int_array_46, full_127)

        # builtin.slice: (-1x2048x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_90 = split_45[0]

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__41 = paddle._C_ops.add_(add__40, slice_90)

        # builtin.slice: (-1x128x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_91 = split_45[1]

        # builtin.combine: ([-1x384x7x7xf16, -1x128x7x7xf16]) <- (-1x384x7x7xf16, -1x128x7x7xf16)
        combine_82 = [concat_80, slice_91]

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x7x7xf16) <- ([-1x384x7x7xf16, -1x128x7x7xf16], 1xi32)
        concat_82 = paddle._C_ops.concat(combine_82, full_128)

        # builtin.combine: ([-1x2048x7x7xf16, -1x512x7x7xf16]) <- (-1x2048x7x7xf16, -1x512x7x7xf16)
        combine_83 = [add__41, concat_82]

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2560x7x7xf16) <- ([-1x2048x7x7xf16, -1x512x7x7xf16], 1xi32)
        concat_83 = paddle._C_ops.concat(combine_83, full_129)

        # pd_op.batch_norm_: (-1x2560x7x7xf16, 2560xf32, 2560xf32, 2560xf32, 2560xf32, None) <- (-1x2560x7x7xf16, 2560xf32, 2560xf32, 2560xf32, 2560xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_83, parameter_655, parameter_656, parameter_657, parameter_658, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x2560x7x7xf16) <- (-1x2560x7x7xf16)
        relu__131 = paddle._C_ops.relu_(batch_norm__786)

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x2560x7x7xf16, 1280x2560x1x1xf16)
        conv2d_131 = paddle._C_ops.conv2d(relu__131, parameter_659, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_660, parameter_661, parameter_662, parameter_663, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu__132 = paddle._C_ops.relu_(batch_norm__792)

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16, 1280x32x3x3xf16)
        conv2d_132 = paddle._C_ops.conv2d(relu__132, parameter_664, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 40, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_665, parameter_666, parameter_667, parameter_668, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu__133 = paddle._C_ops.relu_(batch_norm__798)

        # pd_op.conv2d: (-1x2176x7x7xf16) <- (-1x1280x7x7xf16, 2176x1280x1x1xf16)
        conv2d_133 = paddle._C_ops.conv2d(relu__133, parameter_669, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_47 = [2048, 128]

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x2048x7x7xf16, -1x128x7x7xf16]) <- (-1x2176x7x7xf16, 2xi64, 1xi32)
        split_46 = paddle._C_ops.split(conv2d_133, full_int_array_47, full_130)

        # builtin.slice: (-1x2048x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_92 = split_46[0]

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__42 = paddle._C_ops.add_(add__41, slice_92)

        # builtin.slice: (-1x128x7x7xf16) <- ([-1x2048x7x7xf16, -1x128x7x7xf16])
        slice_93 = split_46[1]

        # builtin.combine: ([-1x512x7x7xf16, -1x128x7x7xf16]) <- (-1x512x7x7xf16, -1x128x7x7xf16)
        combine_84 = [concat_82, slice_93]

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x640x7x7xf16) <- ([-1x512x7x7xf16, -1x128x7x7xf16], 1xi32)
        concat_84 = paddle._C_ops.concat(combine_84, full_131)

        # builtin.combine: ([-1x2048x7x7xf16, -1x640x7x7xf16]) <- (-1x2048x7x7xf16, -1x640x7x7xf16)
        combine_85 = [add__42, concat_84]

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2688x7x7xf16) <- ([-1x2048x7x7xf16, -1x640x7x7xf16], 1xi32)
        concat_85 = paddle._C_ops.concat(combine_85, full_132)

        # pd_op.batch_norm_: (-1x2688x7x7xf16, 2688xf32, 2688xf32, 2688xf32, 2688xf32, None) <- (-1x2688x7x7xf16, 2688xf32, 2688xf32, 2688xf32, 2688xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_85, parameter_670, parameter_671, parameter_672, parameter_673, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x2688x7x7xf16) <- (-1x2688x7x7xf16)
        relu__134 = paddle._C_ops.relu_(batch_norm__804)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_48 = [1, 1]

        # pd_op.pool2d: (-1x2688x1x1xf16) <- (-1x2688x7x7xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__134, full_int_array_48, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x2688xf16, None) <- (-1x2688x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_1, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x2688xf16, 2688x1000xf16)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_674, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__43 = paddle._C_ops.add_(matmul_0, parameter_675)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__43, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__0, paddle.float32)
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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_368, parameter_365, parameter_367, parameter_366, parameter_369, parameter_373, parameter_370, parameter_372, parameter_371, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_388, parameter_385, parameter_387, parameter_386, parameter_389, parameter_393, parameter_390, parameter_392, parameter_391, parameter_394, parameter_398, parameter_395, parameter_397, parameter_396, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_438, parameter_435, parameter_437, parameter_436, parameter_439, parameter_443, parameter_440, parameter_442, parameter_441, parameter_444, parameter_448, parameter_445, parameter_447, parameter_446, parameter_449, parameter_453, parameter_450, parameter_452, parameter_451, parameter_454, parameter_458, parameter_455, parameter_457, parameter_456, parameter_459, parameter_463, parameter_460, parameter_462, parameter_461, parameter_464, parameter_468, parameter_465, parameter_467, parameter_466, parameter_469, parameter_473, parameter_470, parameter_472, parameter_471, parameter_474, parameter_478, parameter_475, parameter_477, parameter_476, parameter_479, parameter_483, parameter_480, parameter_482, parameter_481, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_498, parameter_495, parameter_497, parameter_496, parameter_499, parameter_503, parameter_500, parameter_502, parameter_501, parameter_504, parameter_508, parameter_505, parameter_507, parameter_506, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_533, parameter_530, parameter_532, parameter_531, parameter_534, parameter_538, parameter_535, parameter_537, parameter_536, parameter_539, parameter_543, parameter_540, parameter_542, parameter_541, parameter_544, parameter_548, parameter_545, parameter_547, parameter_546, parameter_549, parameter_553, parameter_550, parameter_552, parameter_551, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_578, parameter_575, parameter_577, parameter_576, parameter_579, parameter_583, parameter_580, parameter_582, parameter_581, parameter_584, parameter_588, parameter_585, parameter_587, parameter_586, parameter_589, parameter_593, parameter_590, parameter_592, parameter_591, parameter_594, parameter_598, parameter_595, parameter_597, parameter_596, parameter_599, parameter_603, parameter_600, parameter_602, parameter_601, parameter_604, parameter_608, parameter_605, parameter_607, parameter_606, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_643, parameter_640, parameter_642, parameter_641, parameter_644, parameter_648, parameter_645, parameter_647, parameter_646, parameter_649, parameter_653, parameter_650, parameter_652, parameter_651, parameter_654, parameter_658, parameter_655, parameter_657, parameter_656, parameter_659, parameter_663, parameter_660, parameter_662, parameter_661, parameter_664, parameter_668, parameter_665, parameter_667, parameter_666, parameter_669, parameter_673, parameter_670, parameter_672, parameter_671, parameter_674, parameter_675, feed_0):
        return self.builtin_module_1633_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_368, parameter_365, parameter_367, parameter_366, parameter_369, parameter_373, parameter_370, parameter_372, parameter_371, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_388, parameter_385, parameter_387, parameter_386, parameter_389, parameter_393, parameter_390, parameter_392, parameter_391, parameter_394, parameter_398, parameter_395, parameter_397, parameter_396, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_428, parameter_425, parameter_427, parameter_426, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_438, parameter_435, parameter_437, parameter_436, parameter_439, parameter_443, parameter_440, parameter_442, parameter_441, parameter_444, parameter_448, parameter_445, parameter_447, parameter_446, parameter_449, parameter_453, parameter_450, parameter_452, parameter_451, parameter_454, parameter_458, parameter_455, parameter_457, parameter_456, parameter_459, parameter_463, parameter_460, parameter_462, parameter_461, parameter_464, parameter_468, parameter_465, parameter_467, parameter_466, parameter_469, parameter_473, parameter_470, parameter_472, parameter_471, parameter_474, parameter_478, parameter_475, parameter_477, parameter_476, parameter_479, parameter_483, parameter_480, parameter_482, parameter_481, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_498, parameter_495, parameter_497, parameter_496, parameter_499, parameter_503, parameter_500, parameter_502, parameter_501, parameter_504, parameter_508, parameter_505, parameter_507, parameter_506, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_533, parameter_530, parameter_532, parameter_531, parameter_534, parameter_538, parameter_535, parameter_537, parameter_536, parameter_539, parameter_543, parameter_540, parameter_542, parameter_541, parameter_544, parameter_548, parameter_545, parameter_547, parameter_546, parameter_549, parameter_553, parameter_550, parameter_552, parameter_551, parameter_554, parameter_558, parameter_555, parameter_557, parameter_556, parameter_559, parameter_563, parameter_560, parameter_562, parameter_561, parameter_564, parameter_568, parameter_565, parameter_567, parameter_566, parameter_569, parameter_573, parameter_570, parameter_572, parameter_571, parameter_574, parameter_578, parameter_575, parameter_577, parameter_576, parameter_579, parameter_583, parameter_580, parameter_582, parameter_581, parameter_584, parameter_588, parameter_585, parameter_587, parameter_586, parameter_589, parameter_593, parameter_590, parameter_592, parameter_591, parameter_594, parameter_598, parameter_595, parameter_597, parameter_596, parameter_599, parameter_603, parameter_600, parameter_602, parameter_601, parameter_604, parameter_608, parameter_605, parameter_607, parameter_606, parameter_609, parameter_613, parameter_610, parameter_612, parameter_611, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_643, parameter_640, parameter_642, parameter_641, parameter_644, parameter_648, parameter_645, parameter_647, parameter_646, parameter_649, parameter_653, parameter_650, parameter_652, parameter_651, parameter_654, parameter_658, parameter_655, parameter_657, parameter_656, parameter_659, parameter_663, parameter_660, parameter_662, parameter_661, parameter_664, parameter_668, parameter_665, parameter_667, parameter_666, parameter_669, parameter_673, parameter_670, parameter_672, parameter_671, parameter_674, parameter_675, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1633_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([128, 3, 7, 7], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([288, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([160, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([160, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([272, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([304], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([304], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([304], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([304], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([160, 304, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([160, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([272, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([160, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([160, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([272, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([336], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([160, 336, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([160, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([272, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([576, 352, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([320, 352, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([320, 608, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([320, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([320, 672, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([320, 704, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([736], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([736], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([736], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([736], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([320, 736, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([320, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([800], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([800], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([800], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([800], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([320, 800, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([320, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([544, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1088, 832, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([640, 832, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([1120], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1120], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([1120], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1120], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([640, 1120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([640, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([1184], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1184], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1184], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1184], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([640, 1184, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1216], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([1216], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([1216], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1216], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([640, 1216, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1248], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1248], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1248], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1248], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([640, 1248, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([640, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1312], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([1312], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([1312], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([1312], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([640, 1312, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1344], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([1344], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1344], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1344], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([640, 1344, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1376], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1376], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1376], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1376], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([640, 1376, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1408], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1408], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1408], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1408], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([640, 1408, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_363
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_368
            paddle.uniform([1440], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([1440], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([1440], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([1440], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([640, 1440, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_378
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_383
            paddle.uniform([1472], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1472], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([1472], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([1472], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([640, 1472, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_388
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_393
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_398
            paddle.uniform([1504], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([1504], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([1504], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([1504], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([640, 1504, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_403
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_408
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_413
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([640, 1536, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_418
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_423
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_428
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([640, 1568, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_433
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_438
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_443
            paddle.uniform([1600], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([1600], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([1600], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([1600], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([640, 1600, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_448
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_453
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_458
            paddle.uniform([1632], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([1632], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([1632], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([1632], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([640, 1632, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_463
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_468
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_473
            paddle.uniform([1664], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([1664], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([1664], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([1664], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([640, 1664, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_478
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_483
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_488
            paddle.uniform([1696], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([1696], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([1696], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([1696], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([640, 1696, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_493
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_498
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_503
            paddle.uniform([1728], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([1728], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([1728], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([1728], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([640, 1728, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_508
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_513
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_518
            paddle.uniform([1760], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([1760], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([1760], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([1760], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([640, 1760, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_523
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_528
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_533
            paddle.uniform([1792], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([1792], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([1792], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([1792], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([640, 1792, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_538
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_543
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_548
            paddle.uniform([1824], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([1824], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([1824], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([1824], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([640, 1824, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_553
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_558
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_563
            paddle.uniform([1856], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([1856], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([1856], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([1856], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([640, 1856, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_568
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_573
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_578
            paddle.uniform([1888], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([1888], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([1888], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([1888], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([640, 1888, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_583
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_588
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_593
            paddle.uniform([1920], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([1920], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([1920], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([1920], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([640, 1920, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_598
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_603
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_604
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_608
            paddle.uniform([1952], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([1952], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([1952], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([1952], dtype='float32', min=0, max=0.5),
            # parameter_609
            paddle.uniform([640, 1952, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_613
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([640, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_618
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_619
            paddle.uniform([1056, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_623
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_624
            paddle.uniform([2304, 1984, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_628
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([1984], dtype='float32', min=0, max=0.5),
            # parameter_629
            paddle.uniform([1280, 1984, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_633
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_634
            paddle.uniform([1280, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_638
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([2176, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_643
            paddle.uniform([2432], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([2432], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([2432], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([2432], dtype='float32', min=0, max=0.5),
            # parameter_644
            paddle.uniform([1280, 2432, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_648
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_649
            paddle.uniform([1280, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_653
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_654
            paddle.uniform([2176, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_658
            paddle.uniform([2560], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([2560], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([2560], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([2560], dtype='float32', min=0, max=0.5),
            # parameter_659
            paddle.uniform([1280, 2560, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_663
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_664
            paddle.uniform([1280, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_668
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_669
            paddle.uniform([2176, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_673
            paddle.uniform([2688], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([2688], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([2688], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([2688], dtype='float32', min=0, max=0.5),
            # parameter_674
            paddle.uniform([2688, 1000], dtype='float16', min=0, max=0.5),
            # parameter_675
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
            paddle.static.InputSpec(shape=[128, 3, 7, 7], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[288, 128, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[160, 128, 1, 1], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[160, 4, 3, 3], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[272, 160, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[304], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[304], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[304], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[304], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[160, 304, 1, 1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[160, 4, 3, 3], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[272, 160, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[160, 320, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[160, 4, 3, 3], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[272, 160, 1, 1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[336], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[336], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[336], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[336], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[160, 336, 1, 1], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[160, 4, 3, 3], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[272, 160, 1, 1], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[576, 352, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[320, 352, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[320, 608, 1, 1], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[320, 640, 1, 1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[320, 672, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[320, 704, 1, 1], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[736], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[736], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[736], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[736], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[320, 736, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[320, 768, 1, 1], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[800], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[800], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[800], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[800], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[320, 800, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[320, 8, 3, 3], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[544, 320, 1, 1], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1088, 832, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[640, 832, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[1120], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[1120], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[1120], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[1120], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[640, 1120, 1, 1], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[640, 1152, 1, 1], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[1184], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[1184], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1184], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1184], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[640, 1184, 1, 1], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[1216], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[1216], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[1216], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[1216], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[640, 1216, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[1248], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[1248], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[1248], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[1248], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[640, 1248, 1, 1], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[640, 1280, 1, 1], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[1312], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[1312], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[1312], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[1312], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[640, 1312, 1, 1], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[1344], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[1344], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[1344], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1344], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[640, 1344, 1, 1], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_338
            paddle.static.InputSpec(shape=[1376], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1376], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1376], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1376], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[640, 1376, 1, 1], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[1408], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1408], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[1408], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[1408], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[640, 1408, 1, 1], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_363
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_368
            paddle.static.InputSpec(shape=[1440], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[1440], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[1440], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[1440], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[640, 1440, 1, 1], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_378
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_383
            paddle.static.InputSpec(shape=[1472], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[1472], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[1472], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[1472], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[640, 1472, 1, 1], dtype='float16'),
            # parameter_388
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_393
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_398
            paddle.static.InputSpec(shape=[1504], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[1504], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[1504], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[1504], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[640, 1504, 1, 1], dtype='float16'),
            # parameter_403
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_408
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_413
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[640, 1536, 1, 1], dtype='float16'),
            # parameter_418
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_423
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_428
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[640, 1568, 1, 1], dtype='float16'),
            # parameter_433
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_438
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_443
            paddle.static.InputSpec(shape=[1600], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[1600], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[1600], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[1600], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[640, 1600, 1, 1], dtype='float16'),
            # parameter_448
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_453
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_458
            paddle.static.InputSpec(shape=[1632], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[1632], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[1632], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[1632], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[640, 1632, 1, 1], dtype='float16'),
            # parameter_463
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_468
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_473
            paddle.static.InputSpec(shape=[1664], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[1664], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[1664], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[1664], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[640, 1664, 1, 1], dtype='float16'),
            # parameter_478
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_483
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_488
            paddle.static.InputSpec(shape=[1696], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[1696], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[1696], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[1696], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[640, 1696, 1, 1], dtype='float16'),
            # parameter_493
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_498
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_503
            paddle.static.InputSpec(shape=[1728], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[1728], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[1728], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[1728], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[640, 1728, 1, 1], dtype='float16'),
            # parameter_508
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_513
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_518
            paddle.static.InputSpec(shape=[1760], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[1760], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[1760], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[1760], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[640, 1760, 1, 1], dtype='float16'),
            # parameter_523
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_528
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_533
            paddle.static.InputSpec(shape=[1792], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[1792], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[1792], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[1792], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[640, 1792, 1, 1], dtype='float16'),
            # parameter_538
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_543
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_548
            paddle.static.InputSpec(shape=[1824], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[1824], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[1824], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[1824], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[640, 1824, 1, 1], dtype='float16'),
            # parameter_553
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_558
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_563
            paddle.static.InputSpec(shape=[1856], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[1856], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[1856], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[1856], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[640, 1856, 1, 1], dtype='float16'),
            # parameter_568
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_573
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_578
            paddle.static.InputSpec(shape=[1888], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[1888], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[1888], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[1888], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[640, 1888, 1, 1], dtype='float16'),
            # parameter_583
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_588
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_593
            paddle.static.InputSpec(shape=[1920], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[1920], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[1920], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[1920], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[640, 1920, 1, 1], dtype='float16'),
            # parameter_598
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_603
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_604
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_608
            paddle.static.InputSpec(shape=[1952], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[1952], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[1952], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[1952], dtype='float32'),
            # parameter_609
            paddle.static.InputSpec(shape=[640, 1952, 1, 1], dtype='float16'),
            # parameter_613
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[640, 16, 3, 3], dtype='float16'),
            # parameter_618
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_619
            paddle.static.InputSpec(shape=[1056, 640, 1, 1], dtype='float16'),
            # parameter_623
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_624
            paddle.static.InputSpec(shape=[2304, 1984, 1, 1], dtype='float16'),
            # parameter_628
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[1984], dtype='float32'),
            # parameter_629
            paddle.static.InputSpec(shape=[1280, 1984, 1, 1], dtype='float16'),
            # parameter_633
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_634
            paddle.static.InputSpec(shape=[1280, 32, 3, 3], dtype='float16'),
            # parameter_638
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[2176, 1280, 1, 1], dtype='float16'),
            # parameter_643
            paddle.static.InputSpec(shape=[2432], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[2432], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[2432], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[2432], dtype='float32'),
            # parameter_644
            paddle.static.InputSpec(shape=[1280, 2432, 1, 1], dtype='float16'),
            # parameter_648
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_649
            paddle.static.InputSpec(shape=[1280, 32, 3, 3], dtype='float16'),
            # parameter_653
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_654
            paddle.static.InputSpec(shape=[2176, 1280, 1, 1], dtype='float16'),
            # parameter_658
            paddle.static.InputSpec(shape=[2560], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[2560], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[2560], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[2560], dtype='float32'),
            # parameter_659
            paddle.static.InputSpec(shape=[1280, 2560, 1, 1], dtype='float16'),
            # parameter_663
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_664
            paddle.static.InputSpec(shape=[1280, 32, 3, 3], dtype='float16'),
            # parameter_668
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_669
            paddle.static.InputSpec(shape=[2176, 1280, 1, 1], dtype='float16'),
            # parameter_673
            paddle.static.InputSpec(shape=[2688], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[2688], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[2688], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[2688], dtype='float32'),
            # parameter_674
            paddle.static.InputSpec(shape=[2688, 1000], dtype='float16'),
            # parameter_675
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