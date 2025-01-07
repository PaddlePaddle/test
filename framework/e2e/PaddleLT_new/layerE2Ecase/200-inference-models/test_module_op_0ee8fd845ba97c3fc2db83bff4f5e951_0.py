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
    return [1897][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2424_0_0(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_46, parameter_47, parameter_49, parameter_48, parameter_50, parameter_51, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_95, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_125, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_185, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_215, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_245, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_305, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_349, parameter_348, parameter_350, parameter_351, parameter_352, parameter_353, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_361, parameter_360, parameter_362, parameter_363, parameter_364, parameter_365, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_373, parameter_372, parameter_374, parameter_375, parameter_376, parameter_377, parameter_379, parameter_378, parameter_380, parameter_381, parameter_382, parameter_383, parameter_385, parameter_384, parameter_386, parameter_387, parameter_388, parameter_389, parameter_391, parameter_390, parameter_392, parameter_393, parameter_394, parameter_395, parameter_397, parameter_396, parameter_398, parameter_399, parameter_400, parameter_401, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_409, parameter_408, parameter_410, parameter_411, parameter_412, parameter_413, parameter_415, parameter_414, parameter_416, parameter_417, parameter_418, parameter_419, parameter_421, parameter_420, parameter_422, parameter_423, parameter_424, parameter_425, parameter_427, parameter_426, parameter_428, parameter_429, parameter_430, parameter_431, parameter_433, parameter_432, parameter_434, parameter_435, parameter_436, parameter_437, parameter_439, parameter_438, parameter_440, parameter_441, parameter_442, parameter_443, parameter_445, parameter_444, parameter_446, parameter_447, parameter_448, parameter_449, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_457, parameter_456, parameter_458, parameter_459, parameter_460, parameter_461, parameter_463, parameter_462, parameter_464, parameter_465, parameter_466, parameter_467, parameter_468, parameter_469, parameter_471, parameter_470, parameter_473, parameter_472, parameter_474, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_481, parameter_480, parameter_482, parameter_483, parameter_484, parameter_485, parameter_486, parameter_487, parameter_489, parameter_488, parameter_490, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_497, parameter_496, parameter_498, parameter_499, parameter_500, parameter_501, parameter_503, parameter_502, parameter_504, parameter_505, parameter_506, parameter_507, parameter_508, parameter_509, parameter_511, parameter_510, parameter_512, parameter_513, parameter_514, parameter_515, parameter_517, parameter_516, parameter_518, parameter_519, feed_0):

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

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x3x224x224xf16, 64x3x4x4xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.flatten_: (-1x64x3136xf16, None) <- (-1x64x56x56xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x64xf16) <- (-1x64x3136xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_0, parameter_4, parameter_5, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x64xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(layer_norm_3, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_6, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__1 = paddle._C_ops.add_(matmul_0, parameter_7)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, full_0, full_1, full_2]

        # pd_op.reshape_: (-1x3136x1x64xf16, 0x-1x3136x64xf16) <- (-1x3136x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x64xf16) <- (-1x3136x1x64xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x3136xf16) <- (-1x3136x64xf16)
        transpose_2 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_3, full_4, full_5]

        # pd_op.reshape_: (-1x64x56x56xf16, 0x-1x64x3136xf16) <- (-1x64x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf16) <- (-1x64x56x56xf16, 64x64x8x8xf16)
        conv2d_1 = paddle._C_ops.conv2d(reshape__2, parameter_8, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x7x7xf16) <- (-1x64x7x7xf16, 1x64x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, full_6, full_7]

        # pd_op.reshape_: (-1x64x49xf16, 0x-1x64x7x7xf16) <- (-1x64x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf16) <- (-1x64x49xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf16, -49xf32, -49xf32) <- (-1x49x64xf16, 64xf32, 64xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_3, parameter_10, parameter_11, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf16) <- (-1x49x64xf16, 64x128xf16)
        matmul_1 = paddle._C_ops.matmul(layer_norm_6, parameter_12, False, False)

        # pd_op.add_: (-1x49x128xf16) <- (-1x49x128xf16, 128xf16)
        add__3 = paddle._C_ops.add_(matmul_1, parameter_13)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, full_8, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x49x2x1x64xf16, 0x-1x49x128xf16) <- (-1x49x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__3, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x64xf16) <- (-1x49x2x1x64xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__6, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_4, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_4, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.transpose: (-1x1x64x49xf16) <- (-1x1x49x64xf16)
        transpose_5 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf16) <- (-1x1x3136x64xf16, -1x1x64x49xf16)
        matmul_2 = paddle._C_ops.matmul(transpose_1, transpose_5, False, False)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_2, full_12, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x1x3136x64xf16) <- (-1x1x3136x49xf16, -1x1x49x64xf16)
        matmul_3 = paddle._C_ops.matmul(softmax__0, slice_3, False, False)

        # pd_op.transpose: (-1x3136x1x64xf16) <- (-1x1x3136x64xf16)
        transpose_6 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, full_13, full_14]

        # pd_op.reshape_: (-1x3136x64xf16, 0x-1x3136x1x64xf16) <- (-1x3136x1x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_4 = paddle._C_ops.matmul(reshape__8, parameter_14, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__4 = paddle._C_ops.add_(matmul_4, parameter_15)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__5 = paddle._C_ops.add_(layer_norm_0, add__4)

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__5, parameter_16, parameter_17, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x512xf16) <- (-1x3136x64xf16, 64x512xf16)
        matmul_5 = paddle._C_ops.matmul(layer_norm_9, parameter_18, False, False)

        # pd_op.add_: (-1x3136x512xf16) <- (-1x3136x512xf16, 512xf16)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_19)

        # pd_op.gelu: (-1x3136x512xf16) <- (-1x3136x512xf16)
        gelu_0 = paddle._C_ops.gelu(add__6, False)

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x512xf16, 512x64xf16)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_20, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__7 = paddle._C_ops.add_(matmul_6, parameter_21)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__8 = paddle._C_ops.add_(add__5, add__7)

        # pd_op.shape: (3xi32) <- (-1x3136x64xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__8, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.transpose: (-1x64x3136xf16) <- (-1x3136x64xf16)
        transpose_7 = paddle._C_ops.transpose(add__8, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_4, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x64x56x56xf16, 0x-1x64x3136xf16) <- (-1x64x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(reshape__10, parameter_22, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_23, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 1x64x1x1xf16)
        add__9 = paddle._C_ops.add_(depthwise_conv2d_0, reshape_4)

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__10 = paddle._C_ops.add_(add__9, reshape__10)

        # pd_op.flatten_: (-1x64x3136xf16, None) <- (-1x64x56x56xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__10, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x3136x64xf16) <- (-1x64x3136xf16)
        transpose_8 = paddle._C_ops.transpose(flatten__2, [0, 2, 1])

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_8, parameter_24, parameter_25, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x64xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(layer_norm_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_3, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_7 = paddle._C_ops.matmul(layer_norm_12, parameter_26, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__11 = paddle._C_ops.add_(matmul_7, parameter_27)

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_5, full_18, full_19, full_20]

        # pd_op.reshape_: (-1x3136x1x64xf16, 0x-1x3136x64xf16) <- (-1x3136x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x64xf16) <- (-1x3136x1x64xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x3136xf16) <- (-1x3136x64xf16)
        transpose_10 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, full_21, full_22, full_23]

        # pd_op.reshape_: (-1x64x56x56xf16, 0x-1x64x3136xf16) <- (-1x64x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf16) <- (-1x64x56x56xf16, 64x64x8x8xf16)
        conv2d_2 = paddle._C_ops.conv2d(reshape__14, parameter_28, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x7x7xf16) <- (-1x64x7x7xf16, 1x64x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_2, reshape_6)

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_5, full_24, full_25]

        # pd_op.reshape_: (-1x64x49xf16, 0x-1x64x7x7xf16) <- (-1x64x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__12, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf16) <- (-1x64x49xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__16, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf16, -49xf32, -49xf32) <- (-1x49x64xf16, 64xf32, 64xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_11, parameter_30, parameter_31, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf16) <- (-1x49x64xf16, 64x128xf16)
        matmul_8 = paddle._C_ops.matmul(layer_norm_15, parameter_32, False, False)

        # pd_op.add_: (-1x49x128xf16) <- (-1x49x128xf16, 128xf16)
        add__13 = paddle._C_ops.add_(matmul_8, parameter_33)

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_5, full_26, full_27, full_28, full_29]

        # pd_op.reshape_: (-1x49x2x1x64xf16, 0x-1x49x128xf16) <- (-1x49x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__13, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x64xf16) <- (-1x49x2x1x64xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__18, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_12, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_12, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.transpose: (-1x1x64x49xf16) <- (-1x1x49x64xf16)
        transpose_13 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf16) <- (-1x1x3136x64xf16, -1x1x64x49xf16)
        matmul_9 = paddle._C_ops.matmul(transpose_9, transpose_13, False, False)

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_9, full_30, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x1x3136x64xf16) <- (-1x1x3136x49xf16, -1x1x49x64xf16)
        matmul_10 = paddle._C_ops.matmul(softmax__1, slice_7, False, False)

        # pd_op.transpose: (-1x3136x1x64xf16) <- (-1x1x3136x64xf16)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_5, full_31, full_32]

        # pd_op.reshape_: (-1x3136x64xf16, 0x-1x3136x1x64xf16) <- (-1x3136x1x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_11 = paddle._C_ops.matmul(reshape__20, parameter_34, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__14 = paddle._C_ops.add_(matmul_11, parameter_35)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__15 = paddle._C_ops.add_(transpose_8, add__14)

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__15, parameter_36, parameter_37, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x512xf16) <- (-1x3136x64xf16, 64x512xf16)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_38, False, False)

        # pd_op.add_: (-1x3136x512xf16) <- (-1x3136x512xf16, 512xf16)
        add__16 = paddle._C_ops.add_(matmul_12, parameter_39)

        # pd_op.gelu: (-1x3136x512xf16) <- (-1x3136x512xf16)
        gelu_1 = paddle._C_ops.gelu(add__16, False)

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x512xf16, 512x64xf16)
        matmul_13 = paddle._C_ops.matmul(gelu_1, parameter_40, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__17 = paddle._C_ops.add_(matmul_13, parameter_41)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__18 = paddle._C_ops.add_(add__15, add__17)

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__18, parameter_42, parameter_43, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x3136x64xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(layer_norm_21, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_4, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_14 = paddle._C_ops.matmul(layer_norm_21, parameter_44, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__19 = paddle._C_ops.add_(matmul_14, parameter_45)

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_8, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x3136x1x64xf16, 0x-1x3136x64xf16) <- (-1x3136x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__19, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x3136x64xf16) <- (-1x3136x1x64xf16)
        transpose_15 = paddle._C_ops.transpose(reshape__22, [0, 2, 1, 3])

        # pd_op.transpose: (-1x64x3136xf16) <- (-1x3136x64xf16)
        transpose_16 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_8, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x64x56x56xf16, 0x-1x64x3136xf16) <- (-1x64x3136xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x7x7xf16) <- (-1x64x56x56xf16, 64x64x8x8xf16)
        conv2d_3 = paddle._C_ops.conv2d(reshape__24, parameter_46, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_47, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x7x7xf16) <- (-1x64x7x7xf16, 1x64x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_3, reshape_8)

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_13 = [slice_8, full_39, full_40]

        # pd_op.reshape_: (-1x64x49xf16, 0x-1x64x7x7xf16) <- (-1x64x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x64xf16) <- (-1x64x49xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__26, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x64xf16, -49xf32, -49xf32) <- (-1x49x64xf16, 64xf32, 64xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_17, parameter_48, parameter_49, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x128xf16) <- (-1x49x64xf16, 64x128xf16)
        matmul_15 = paddle._C_ops.matmul(layer_norm_24, parameter_50, False, False)

        # pd_op.add_: (-1x49x128xf16) <- (-1x49x128xf16, 128xf16)
        add__21 = paddle._C_ops.add_(matmul_15, parameter_51)

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_8, full_41, full_42, full_43, full_44]

        # pd_op.reshape_: (-1x49x2x1x64xf16, 0x-1x49x128xf16) <- (-1x49x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__21, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x1x49x64xf16) <- (-1x49x2x1x64xf16)
        transpose_18 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_18, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [2]

        # pd_op.slice: (-1x1x49x64xf16) <- (2x-1x1x49x64xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_18, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.transpose: (-1x1x64x49xf16) <- (-1x1x49x64xf16)
        transpose_19 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])

        # pd_op.matmul: (-1x1x3136x49xf16) <- (-1x1x3136x64xf16, -1x1x64x49xf16)
        matmul_16 = paddle._C_ops.matmul(transpose_15, transpose_19, False, False)

        # pd_op.full: (1xf32) <- ()
        full_45 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_16, full_45, float('0'), True)

        # pd_op.softmax_: (-1x1x3136x49xf16) <- (-1x1x3136x49xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x1x3136x64xf16) <- (-1x1x3136x49xf16, -1x1x49x64xf16)
        matmul_17 = paddle._C_ops.matmul(softmax__2, slice_10, False, False)

        # pd_op.transpose: (-1x3136x1x64xf16) <- (-1x1x3136x64xf16)
        transpose_20 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('3136'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_8, full_46, full_47]

        # pd_op.reshape_: (-1x3136x64xf16, 0x-1x3136x1x64xf16) <- (-1x3136x1x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x64xf16, 64x64xf16)
        matmul_18 = paddle._C_ops.matmul(reshape__30, parameter_52, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__22 = paddle._C_ops.add_(matmul_18, parameter_53)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__23 = paddle._C_ops.add_(add__18, add__22)

        # pd_op.layer_norm: (-1x3136x64xf16, -3136xf32, -3136xf32) <- (-1x3136x64xf16, 64xf32, 64xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__23, parameter_54, parameter_55, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x3136x512xf16) <- (-1x3136x64xf16, 64x512xf16)
        matmul_19 = paddle._C_ops.matmul(layer_norm_27, parameter_56, False, False)

        # pd_op.add_: (-1x3136x512xf16) <- (-1x3136x512xf16, 512xf16)
        add__24 = paddle._C_ops.add_(matmul_19, parameter_57)

        # pd_op.gelu: (-1x3136x512xf16) <- (-1x3136x512xf16)
        gelu_2 = paddle._C_ops.gelu(add__24, False)

        # pd_op.matmul: (-1x3136x64xf16) <- (-1x3136x512xf16, 512x64xf16)
        matmul_20 = paddle._C_ops.matmul(gelu_2, parameter_58, False, False)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, 64xf16)
        add__25 = paddle._C_ops.add_(matmul_20, parameter_59)

        # pd_op.add_: (-1x3136x64xf16) <- (-1x3136x64xf16, -1x3136x64xf16)
        add__26 = paddle._C_ops.add_(add__23, add__25)

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_0, full_48, full_49, full_50]

        # pd_op.reshape_: (-1x56x56x64xf16, 0x-1x3136x64xf16) <- (-1x3136x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x64x56x56xf16) <- (-1x56x56x64xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__32, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x64x56x56xf16, 128x64x2x2xf16)
        conv2d_4 = paddle._C_ops.conv2d(transpose_21, parameter_60, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_61, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__27 = paddle._C_ops.add_(conv2d_4, reshape_10)

        # pd_op.flatten_: (-1x128x784xf16, None) <- (-1x128x28x28xf16)
        flatten__4, flatten__5 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__27, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x128xf16) <- (-1x128x784xf16)
        transpose_22 = paddle._C_ops.transpose(flatten__4, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_22, parameter_62, parameter_63, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_30, parameter_64, parameter_65, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x128xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(layer_norm_33, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_5, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_21 = paddle._C_ops.matmul(layer_norm_33, parameter_66, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__28 = paddle._C_ops.add_(matmul_21, parameter_67)

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_11, full_51, full_52, full_53]

        # pd_op.reshape_: (-1x784x2x64xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__28, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x64xf16) <- (-1x784x2x64xf16)
        transpose_23 = paddle._C_ops.transpose(reshape__34, [0, 2, 1, 3])

        # pd_op.transpose: (-1x128x784xf16) <- (-1x784x128xf16)
        transpose_24 = paddle._C_ops.transpose(layer_norm_33, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_11, full_54, full_55, full_56]

        # pd_op.reshape_: (-1x128x28x28xf16, 0x-1x128x784xf16) <- (-1x128x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x7x7xf16) <- (-1x128x28x28xf16, 128x128x4x4xf16)
        conv2d_5 = paddle._C_ops.conv2d(reshape__36, parameter_68, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x7x7xf16) <- (-1x128x7x7xf16, 1x128x1x1xf16)
        add__29 = paddle._C_ops.add_(conv2d_5, reshape_12)

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_11, full_57, full_58]

        # pd_op.reshape_: (-1x128x49xf16, 0x-1x128x7x7xf16) <- (-1x128x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__29, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x128xf16) <- (-1x128x49xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__38, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x128xf16, -49xf32, -49xf32) <- (-1x49x128xf16, 128xf32, 128xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_25, parameter_70, parameter_71, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x128xf16, 128x256xf16)
        matmul_22 = paddle._C_ops.matmul(layer_norm_36, parameter_72, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__30 = paddle._C_ops.add_(matmul_22, parameter_73)

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_11, full_59, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x49x2x2x64xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x64xf16) <- (-1x49x2x2x64xf16)
        transpose_26 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [1]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(transpose_26, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [2]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_26, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.transpose: (-1x2x64x49xf16) <- (-1x2x49x64xf16)
        transpose_27 = paddle._C_ops.transpose(slice_12, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x64xf16, -1x2x64x49xf16)
        matmul_23 = paddle._C_ops.matmul(transpose_23, transpose_27, False, False)

        # pd_op.full: (1xf32) <- ()
        full_63 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_23, full_63, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x2x784x64xf16) <- (-1x2x784x49xf16, -1x2x49x64xf16)
        matmul_24 = paddle._C_ops.matmul(softmax__3, slice_13, False, False)

        # pd_op.transpose: (-1x784x2x64xf16) <- (-1x2x784x64xf16)
        transpose_28 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_11, full_64, full_65]

        # pd_op.reshape_: (-1x784x128xf16, 0x-1x784x2x64xf16) <- (-1x784x2x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_25 = paddle._C_ops.matmul(reshape__42, parameter_74, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__31 = paddle._C_ops.add_(matmul_25, parameter_75)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__32 = paddle._C_ops.add_(layer_norm_30, add__31)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__32, parameter_76, parameter_77, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x1024xf16) <- (-1x784x128xf16, 128x1024xf16)
        matmul_26 = paddle._C_ops.matmul(layer_norm_39, parameter_78, False, False)

        # pd_op.add_: (-1x784x1024xf16) <- (-1x784x1024xf16, 1024xf16)
        add__33 = paddle._C_ops.add_(matmul_26, parameter_79)

        # pd_op.gelu: (-1x784x1024xf16) <- (-1x784x1024xf16)
        gelu_3 = paddle._C_ops.gelu(add__33, False)

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x1024xf16, 1024x128xf16)
        matmul_27 = paddle._C_ops.matmul(gelu_3, parameter_80, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__34 = paddle._C_ops.add_(matmul_27, parameter_81)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__35 = paddle._C_ops.add_(add__32, add__34)

        # pd_op.shape: (3xi32) <- (-1x784x128xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(add__35, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_6, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.transpose: (-1x128x784xf16) <- (-1x784x128xf16)
        transpose_29 = paddle._C_ops.transpose(add__35, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_14, full_66, full_67, full_68]

        # pd_op.reshape_: (-1x128x28x28xf16, 0x-1x128x784xf16) <- (-1x128x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 128x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(reshape__44, parameter_82, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_83, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, 1x128x1x1xf16)
        add__36 = paddle._C_ops.add_(depthwise_conv2d_1, reshape_14)

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__37 = paddle._C_ops.add_(add__36, reshape__44)

        # pd_op.flatten_: (-1x128x784xf16, None) <- (-1x128x28x28xf16)
        flatten__6, flatten__7 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__37, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x128xf16) <- (-1x128x784xf16)
        transpose_30 = paddle._C_ops.transpose(flatten__6, [0, 2, 1])

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_30, parameter_84, parameter_85, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x128xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(layer_norm_42, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_7, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_28 = paddle._C_ops.matmul(layer_norm_42, parameter_86, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__38 = paddle._C_ops.add_(matmul_28, parameter_87)

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_15, full_69, full_70, full_71]

        # pd_op.reshape_: (-1x784x2x64xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x64xf16) <- (-1x784x2x64xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__46, [0, 2, 1, 3])

        # pd_op.transpose: (-1x128x784xf16) <- (-1x784x128xf16)
        transpose_32 = paddle._C_ops.transpose(layer_norm_42, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_15, full_72, full_73, full_74]

        # pd_op.reshape_: (-1x128x28x28xf16, 0x-1x128x784xf16) <- (-1x128x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_32, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x7x7xf16) <- (-1x128x28x28xf16, 128x128x4x4xf16)
        conv2d_6 = paddle._C_ops.conv2d(reshape__48, parameter_88, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x7x7xf16) <- (-1x128x7x7xf16, 1x128x1x1xf16)
        add__39 = paddle._C_ops.add_(conv2d_6, reshape_16)

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_15, full_75, full_76]

        # pd_op.reshape_: (-1x128x49xf16, 0x-1x128x7x7xf16) <- (-1x128x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x128xf16) <- (-1x128x49xf16)
        transpose_33 = paddle._C_ops.transpose(reshape__50, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x128xf16, -49xf32, -49xf32) <- (-1x49x128xf16, 128xf32, 128xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_33, parameter_90, parameter_91, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x128xf16, 128x256xf16)
        matmul_29 = paddle._C_ops.matmul(layer_norm_45, parameter_92, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__40 = paddle._C_ops.add_(matmul_29, parameter_93)

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_15, full_77, full_78, full_79, full_80]

        # pd_op.reshape_: (-1x49x2x2x64xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__40, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x64xf16) <- (-1x49x2x2x64xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__52, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_34, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [2]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_34, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.transpose: (-1x2x64x49xf16) <- (-1x2x49x64xf16)
        transpose_35 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x64xf16, -1x2x64x49xf16)
        matmul_30 = paddle._C_ops.matmul(transpose_31, transpose_35, False, False)

        # pd_op.full: (1xf32) <- ()
        full_81 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_30, full_81, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x2x784x64xf16) <- (-1x2x784x49xf16, -1x2x49x64xf16)
        matmul_31 = paddle._C_ops.matmul(softmax__4, slice_17, False, False)

        # pd_op.transpose: (-1x784x2x64xf16) <- (-1x2x784x64xf16)
        transpose_36 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_27 = [slice_15, full_82, full_83]

        # pd_op.reshape_: (-1x784x128xf16, 0x-1x784x2x64xf16) <- (-1x784x2x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, [x.reshape([]) for x in combine_27]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_32 = paddle._C_ops.matmul(reshape__54, parameter_94, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__41 = paddle._C_ops.add_(matmul_32, parameter_95)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__42 = paddle._C_ops.add_(transpose_30, add__41)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__42, parameter_96, parameter_97, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x1024xf16) <- (-1x784x128xf16, 128x1024xf16)
        matmul_33 = paddle._C_ops.matmul(layer_norm_48, parameter_98, False, False)

        # pd_op.add_: (-1x784x1024xf16) <- (-1x784x1024xf16, 1024xf16)
        add__43 = paddle._C_ops.add_(matmul_33, parameter_99)

        # pd_op.gelu: (-1x784x1024xf16) <- (-1x784x1024xf16)
        gelu_4 = paddle._C_ops.gelu(add__43, False)

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x1024xf16, 1024x128xf16)
        matmul_34 = paddle._C_ops.matmul(gelu_4, parameter_100, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__44 = paddle._C_ops.add_(matmul_34, parameter_101)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__45 = paddle._C_ops.add_(add__42, add__44)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__45, parameter_102, parameter_103, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x128xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(layer_norm_51, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_8, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_35 = paddle._C_ops.matmul(layer_norm_51, parameter_104, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__46 = paddle._C_ops.add_(matmul_35, parameter_105)

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_18, full_84, full_85, full_86]

        # pd_op.reshape_: (-1x784x2x64xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__46, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x64xf16) <- (-1x784x2x64xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__56, [0, 2, 1, 3])

        # pd_op.transpose: (-1x128x784xf16) <- (-1x784x128xf16)
        transpose_38 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_18, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x128x28x28xf16, 0x-1x128x784xf16) <- (-1x128x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x7x7xf16) <- (-1x128x28x28xf16, 128x128x4x4xf16)
        conv2d_7 = paddle._C_ops.conv2d(reshape__58, parameter_106, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_107, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x7x7xf16) <- (-1x128x7x7xf16, 1x128x1x1xf16)
        add__47 = paddle._C_ops.add_(conv2d_7, reshape_18)

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_18, full_90, full_91]

        # pd_op.reshape_: (-1x128x49xf16, 0x-1x128x7x7xf16) <- (-1x128x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__47, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x128xf16) <- (-1x128x49xf16)
        transpose_39 = paddle._C_ops.transpose(reshape__60, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x128xf16, -49xf32, -49xf32) <- (-1x49x128xf16, 128xf32, 128xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_39, parameter_108, parameter_109, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x128xf16, 128x256xf16)
        matmul_36 = paddle._C_ops.matmul(layer_norm_54, parameter_110, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__48 = paddle._C_ops.add_(matmul_36, parameter_111)

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_18, full_92, full_93, full_94, full_95]

        # pd_op.reshape_: (-1x49x2x2x64xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x64xf16) <- (-1x49x2x2x64xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__62, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_40, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [2]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(transpose_40, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.transpose: (-1x2x64x49xf16) <- (-1x2x49x64xf16)
        transpose_41 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x64xf16, -1x2x64x49xf16)
        matmul_37 = paddle._C_ops.matmul(transpose_37, transpose_41, False, False)

        # pd_op.full: (1xf32) <- ()
        full_96 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_37, full_96, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x2x784x64xf16) <- (-1x2x784x49xf16, -1x2x49x64xf16)
        matmul_38 = paddle._C_ops.matmul(softmax__5, slice_20, False, False)

        # pd_op.transpose: (-1x784x2x64xf16) <- (-1x2x784x64xf16)
        transpose_42 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_18, full_97, full_98]

        # pd_op.reshape_: (-1x784x128xf16, 0x-1x784x2x64xf16) <- (-1x784x2x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_39 = paddle._C_ops.matmul(reshape__64, parameter_112, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__49 = paddle._C_ops.add_(matmul_39, parameter_113)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__50 = paddle._C_ops.add_(add__45, add__49)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__50, parameter_114, parameter_115, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x1024xf16) <- (-1x784x128xf16, 128x1024xf16)
        matmul_40 = paddle._C_ops.matmul(layer_norm_57, parameter_116, False, False)

        # pd_op.add_: (-1x784x1024xf16) <- (-1x784x1024xf16, 1024xf16)
        add__51 = paddle._C_ops.add_(matmul_40, parameter_117)

        # pd_op.gelu: (-1x784x1024xf16) <- (-1x784x1024xf16)
        gelu_5 = paddle._C_ops.gelu(add__51, False)

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x1024xf16, 1024x128xf16)
        matmul_41 = paddle._C_ops.matmul(gelu_5, parameter_118, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__52 = paddle._C_ops.add_(matmul_41, parameter_119)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__53 = paddle._C_ops.add_(add__50, add__52)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__53, parameter_120, parameter_121, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x784x128xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(layer_norm_60, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_9, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_42 = paddle._C_ops.matmul(layer_norm_60, parameter_122, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__54 = paddle._C_ops.add_(matmul_42, parameter_123)

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_21, full_99, full_100, full_101]

        # pd_op.reshape_: (-1x784x2x64xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__54, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x784x64xf16) <- (-1x784x2x64xf16)
        transpose_43 = paddle._C_ops.transpose(reshape__66, [0, 2, 1, 3])

        # pd_op.transpose: (-1x128x784xf16) <- (-1x784x128xf16)
        transpose_44 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_21, full_102, full_103, full_104]

        # pd_op.reshape_: (-1x128x28x28xf16, 0x-1x128x784xf16) <- (-1x128x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_44, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x7x7xf16) <- (-1x128x28x28xf16, 128x128x4x4xf16)
        conv2d_8 = paddle._C_ops.conv2d(reshape__68, parameter_124, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_125, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x7x7xf16) <- (-1x128x7x7xf16, 1x128x1x1xf16)
        add__55 = paddle._C_ops.add_(conv2d_8, reshape_20)

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_35 = [slice_21, full_105, full_106]

        # pd_op.reshape_: (-1x128x49xf16, 0x-1x128x7x7xf16) <- (-1x128x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__55, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x128xf16) <- (-1x128x49xf16)
        transpose_45 = paddle._C_ops.transpose(reshape__70, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x128xf16, -49xf32, -49xf32) <- (-1x49x128xf16, 128xf32, 128xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_45, parameter_126, parameter_127, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x256xf16) <- (-1x49x128xf16, 128x256xf16)
        matmul_43 = paddle._C_ops.matmul(layer_norm_63, parameter_128, False, False)

        # pd_op.add_: (-1x49x256xf16) <- (-1x49x256xf16, 256xf16)
        add__56 = paddle._C_ops.add_(matmul_43, parameter_129)

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_36 = [slice_21, full_107, full_108, full_109, full_110]

        # pd_op.reshape_: (-1x49x2x2x64xf16, 0x-1x49x256xf16) <- (-1x49x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x2x49x64xf16) <- (-1x49x2x2x64xf16)
        transpose_46 = paddle._C_ops.transpose(reshape__72, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_46, [0], full_int_array_55, full_int_array_56, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.slice: (-1x2x49x64xf16) <- (2x-1x2x49x64xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_46, [0], full_int_array_57, full_int_array_58, [1], [0])

        # pd_op.transpose: (-1x2x64x49xf16) <- (-1x2x49x64xf16)
        transpose_47 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x2x784x49xf16) <- (-1x2x784x64xf16, -1x2x64x49xf16)
        matmul_44 = paddle._C_ops.matmul(transpose_43, transpose_47, False, False)

        # pd_op.full: (1xf32) <- ()
        full_111 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x2x784x49xf16) <- (-1x2x784x49xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_44, full_111, float('0'), True)

        # pd_op.softmax_: (-1x2x784x49xf16) <- (-1x2x784x49xf16)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x2x784x64xf16) <- (-1x2x784x49xf16, -1x2x49x64xf16)
        matmul_45 = paddle._C_ops.matmul(softmax__6, slice_23, False, False)

        # pd_op.transpose: (-1x784x2x64xf16) <- (-1x2x784x64xf16)
        transpose_48 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_37 = [slice_21, full_112, full_113]

        # pd_op.reshape_: (-1x784x128xf16, 0x-1x784x2x64xf16) <- (-1x784x2x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_48, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x128xf16, 128x128xf16)
        matmul_46 = paddle._C_ops.matmul(reshape__74, parameter_130, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__57 = paddle._C_ops.add_(matmul_46, parameter_131)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__58 = paddle._C_ops.add_(add__53, add__57)

        # pd_op.layer_norm: (-1x784x128xf16, -784xf32, -784xf32) <- (-1x784x128xf16, 128xf32, 128xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__58, parameter_132, parameter_133, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x784x1024xf16) <- (-1x784x128xf16, 128x1024xf16)
        matmul_47 = paddle._C_ops.matmul(layer_norm_66, parameter_134, False, False)

        # pd_op.add_: (-1x784x1024xf16) <- (-1x784x1024xf16, 1024xf16)
        add__59 = paddle._C_ops.add_(matmul_47, parameter_135)

        # pd_op.gelu: (-1x784x1024xf16) <- (-1x784x1024xf16)
        gelu_6 = paddle._C_ops.gelu(add__59, False)

        # pd_op.matmul: (-1x784x128xf16) <- (-1x784x1024xf16, 1024x128xf16)
        matmul_48 = paddle._C_ops.matmul(gelu_6, parameter_136, False, False)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, 128xf16)
        add__60 = paddle._C_ops.add_(matmul_48, parameter_137)

        # pd_op.add_: (-1x784x128xf16) <- (-1x784x128xf16, -1x784x128xf16)
        add__61 = paddle._C_ops.add_(add__58, add__60)

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_0, full_114, full_115, full_116]

        # pd_op.reshape_: (-1x28x28x128xf16, 0x-1x784x128xf16) <- (-1x784x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__61, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x128x28x28xf16) <- (-1x28x28x128xf16)
        transpose_49 = paddle._C_ops.transpose(reshape__76, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x320x14x14xf16) <- (-1x128x28x28xf16, 320x128x2x2xf16)
        conv2d_9 = paddle._C_ops.conv2d(transpose_49, parameter_138, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__62 = paddle._C_ops.add_(conv2d_9, reshape_22)

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__8, flatten__9 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__62, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_50 = paddle._C_ops.transpose(flatten__8, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_50, parameter_140, parameter_141, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_69, parameter_142, parameter_143, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(layer_norm_72, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_10, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_49 = paddle._C_ops.matmul(layer_norm_72, parameter_144, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__63 = paddle._C_ops.add_(matmul_49, parameter_145)

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_24, full_117, full_118, full_119]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__63, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_51 = paddle._C_ops.transpose(reshape__78, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_52 = paddle._C_ops.transpose(layer_norm_72, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_24, full_120, full_121, full_122]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_52, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_10 = paddle._C_ops.conv2d(reshape__80, parameter_146, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_147, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__64 = paddle._C_ops.add_(conv2d_10, reshape_24)

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_41 = [slice_24, full_123, full_124]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__64, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_53 = paddle._C_ops.transpose(reshape__82, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_53, parameter_148, parameter_149, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_50 = paddle._C_ops.matmul(layer_norm_75, parameter_150, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__65 = paddle._C_ops.add_(matmul_50, parameter_151)

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_24, full_125, full_126, full_127, full_128]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__65, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_54 = paddle._C_ops.transpose(reshape__84, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_54, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_54, [0], full_int_array_65, full_int_array_66, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_55 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_51 = paddle._C_ops.matmul(transpose_51, transpose_55, False, False)

        # pd_op.full: (1xf32) <- ()
        full_129 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_51, full_129, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_52 = paddle._C_ops.matmul(softmax__7, slice_26, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_56 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_24, full_130, full_131]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_56, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_53 = paddle._C_ops.matmul(reshape__86, parameter_152, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__66 = paddle._C_ops.add_(matmul_53, parameter_153)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__67 = paddle._C_ops.add_(layer_norm_69, add__66)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__67, parameter_154, parameter_155, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_54 = paddle._C_ops.matmul(layer_norm_78, parameter_156, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__68 = paddle._C_ops.add_(matmul_54, parameter_157)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_7 = paddle._C_ops.gelu(add__68, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_55 = paddle._C_ops.matmul(gelu_7, parameter_158, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__69 = paddle._C_ops.add_(matmul_55, parameter_159)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__70 = paddle._C_ops.add_(add__67, add__69)

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(add__70, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_11, [0], full_int_array_67, full_int_array_68, [1], [0])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_57 = paddle._C_ops.transpose(add__70, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_27, full_132, full_133, full_134]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_57, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 320x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(reshape__88, parameter_160, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, 1x320x1x1xf16)
        add__71 = paddle._C_ops.add_(depthwise_conv2d_2, reshape_26)

        # pd_op.add_: (-1x320x14x14xf16) <- (-1x320x14x14xf16, -1x320x14x14xf16)
        add__72 = paddle._C_ops.add_(add__71, reshape__88)

        # pd_op.flatten_: (-1x320x196xf16, None) <- (-1x320x14x14xf16)
        flatten__10, flatten__11 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__72, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x320xf16) <- (-1x320x196xf16)
        transpose_58 = paddle._C_ops.transpose(flatten__10, [0, 2, 1])

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_58, parameter_162, parameter_163, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(layer_norm_81, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_12, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_56 = paddle._C_ops.matmul(layer_norm_81, parameter_164, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__73 = paddle._C_ops.add_(matmul_56, parameter_165)

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_28, full_135, full_136, full_137]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__73, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_59 = paddle._C_ops.transpose(reshape__90, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_60 = paddle._C_ops.transpose(layer_norm_81, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_28, full_138, full_139, full_140]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_60, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_11 = paddle._C_ops.conv2d(reshape__92, parameter_166, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_72 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__74 = paddle._C_ops.add_(conv2d_11, reshape_28)

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_28, full_141, full_142]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__74, [x.reshape([]) for x in combine_47]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_61 = paddle._C_ops.transpose(reshape__94, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_61, parameter_168, parameter_169, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_57 = paddle._C_ops.matmul(layer_norm_84, parameter_170, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__75 = paddle._C_ops.add_(matmul_57, parameter_171)

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_28, full_143, full_144, full_145, full_146]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__75, [x.reshape([]) for x in combine_48]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__96, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_62, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_62, [0], full_int_array_75, full_int_array_76, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_63 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_58 = paddle._C_ops.matmul(transpose_59, transpose_63, False, False)

        # pd_op.full: (1xf32) <- ()
        full_147 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_58, full_147, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__8 = paddle._C_ops.softmax_(scale__8, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_59 = paddle._C_ops.matmul(softmax__8, slice_30, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_64 = paddle._C_ops.transpose(matmul_59, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_49 = [slice_28, full_148, full_149]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_64, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_60 = paddle._C_ops.matmul(reshape__98, parameter_172, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__76 = paddle._C_ops.add_(matmul_60, parameter_173)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__77 = paddle._C_ops.add_(transpose_58, add__76)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__77, parameter_174, parameter_175, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_61 = paddle._C_ops.matmul(layer_norm_87, parameter_176, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__78 = paddle._C_ops.add_(matmul_61, parameter_177)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_8 = paddle._C_ops.gelu(add__78, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_62 = paddle._C_ops.matmul(gelu_8, parameter_178, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__79 = paddle._C_ops.add_(matmul_62, parameter_179)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__80 = paddle._C_ops.add_(add__77, add__79)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__80, parameter_180, parameter_181, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(layer_norm_90, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(shape_13, [0], full_int_array_77, full_int_array_78, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_63 = paddle._C_ops.matmul(layer_norm_90, parameter_182, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__81 = paddle._C_ops.add_(matmul_63, parameter_183)

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_31, full_150, full_151, full_152]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__81, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_65 = paddle._C_ops.transpose(reshape__100, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_66 = paddle._C_ops.transpose(layer_norm_90, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_31, full_153, full_154, full_155]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_66, [x.reshape([]) for x in combine_51]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_12 = paddle._C_ops.conv2d(reshape__102, parameter_184, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__82 = paddle._C_ops.add_(conv2d_12, reshape_30)

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_52 = [slice_31, full_156, full_157]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__82, [x.reshape([]) for x in combine_52]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_67 = paddle._C_ops.transpose(reshape__104, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_67, parameter_186, parameter_187, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_64 = paddle._C_ops.matmul(layer_norm_93, parameter_188, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__83 = paddle._C_ops.add_(matmul_64, parameter_189)

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_31, full_158, full_159, full_160, full_161]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__83, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_68 = paddle._C_ops.transpose(reshape__106, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_68, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_68, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_69 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_65 = paddle._C_ops.matmul(transpose_65, transpose_69, False, False)

        # pd_op.full: (1xf32) <- ()
        full_162 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_65, full_162, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__9 = paddle._C_ops.softmax_(scale__9, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_66 = paddle._C_ops.matmul(softmax__9, slice_33, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_70 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_54 = [slice_31, full_163, full_164]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_70, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_67 = paddle._C_ops.matmul(reshape__108, parameter_190, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__84 = paddle._C_ops.add_(matmul_67, parameter_191)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__85 = paddle._C_ops.add_(add__80, add__84)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__85, parameter_192, parameter_193, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_68 = paddle._C_ops.matmul(layer_norm_96, parameter_194, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__86 = paddle._C_ops.add_(matmul_68, parameter_195)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_9 = paddle._C_ops.gelu(add__86, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_69 = paddle._C_ops.matmul(gelu_9, parameter_196, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__87 = paddle._C_ops.add_(matmul_69, parameter_197)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__88 = paddle._C_ops.add_(add__85, add__87)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__88, parameter_198, parameter_199, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(layer_norm_99, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(shape_14, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_70 = paddle._C_ops.matmul(layer_norm_99, parameter_200, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__89 = paddle._C_ops.add_(matmul_70, parameter_201)

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_34, full_165, full_166, full_167]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__89, [x.reshape([]) for x in combine_55]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_71 = paddle._C_ops.transpose(reshape__110, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_72 = paddle._C_ops.transpose(layer_norm_99, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [slice_34, full_168, full_169, full_170]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_72, [x.reshape([]) for x in combine_56]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_13 = paddle._C_ops.conv2d(reshape__112, parameter_202, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_203, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__90 = paddle._C_ops.add_(conv2d_13, reshape_32)

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_57 = [slice_34, full_171, full_172]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__90, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_73 = paddle._C_ops.transpose(reshape__114, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_73, parameter_204, parameter_205, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_71 = paddle._C_ops.matmul(layer_norm_102, parameter_206, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__91 = paddle._C_ops.add_(matmul_71, parameter_207)

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_34, full_173, full_174, full_175, full_176]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__91, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_74 = paddle._C_ops.transpose(reshape__116, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_74, [0], full_int_array_87, full_int_array_88, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_74, [0], full_int_array_89, full_int_array_90, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_75 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_72 = paddle._C_ops.matmul(transpose_71, transpose_75, False, False)

        # pd_op.full: (1xf32) <- ()
        full_177 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_72, full_177, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__10 = paddle._C_ops.softmax_(scale__10, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_73 = paddle._C_ops.matmul(softmax__10, slice_36, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_76 = paddle._C_ops.transpose(matmul_73, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_59 = [slice_34, full_178, full_179]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_76, [x.reshape([]) for x in combine_59]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_74 = paddle._C_ops.matmul(reshape__118, parameter_208, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__92 = paddle._C_ops.add_(matmul_74, parameter_209)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__93 = paddle._C_ops.add_(add__88, add__92)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__93, parameter_210, parameter_211, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_75 = paddle._C_ops.matmul(layer_norm_105, parameter_212, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__94 = paddle._C_ops.add_(matmul_75, parameter_213)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_10 = paddle._C_ops.gelu(add__94, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_76 = paddle._C_ops.matmul(gelu_10, parameter_214, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__95 = paddle._C_ops.add_(matmul_76, parameter_215)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__96 = paddle._C_ops.add_(add__93, add__95)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__96, parameter_216, parameter_217, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(layer_norm_108, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(shape_15, [0], full_int_array_91, full_int_array_92, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_77 = paddle._C_ops.matmul(layer_norm_108, parameter_218, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__97 = paddle._C_ops.add_(matmul_77, parameter_219)

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_182 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_60 = [slice_37, full_180, full_181, full_182]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__97, [x.reshape([]) for x in combine_60]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_77 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_78 = paddle._C_ops.transpose(layer_norm_108, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_37, full_183, full_184, full_185]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_78, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_14 = paddle._C_ops.conv2d(reshape__122, parameter_220, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_93 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_221, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__98 = paddle._C_ops.add_(conv2d_14, reshape_34)

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_62 = [slice_37, full_186, full_187]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__98, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_79 = paddle._C_ops.transpose(reshape__124, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_79, parameter_222, parameter_223, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_78 = paddle._C_ops.matmul(layer_norm_111, parameter_224, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__99 = paddle._C_ops.add_(matmul_78, parameter_225)

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_37, full_188, full_189, full_190, full_191]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__99, [x.reshape([]) for x in combine_63]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_80 = paddle._C_ops.transpose(reshape__126, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(transpose_80, [0], full_int_array_94, full_int_array_95, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_80, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_81 = paddle._C_ops.transpose(slice_38, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_79 = paddle._C_ops.matmul(transpose_77, transpose_81, False, False)

        # pd_op.full: (1xf32) <- ()
        full_192 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_79, full_192, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__11 = paddle._C_ops.softmax_(scale__11, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_80 = paddle._C_ops.matmul(softmax__11, slice_39, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_82 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_64 = [slice_37, full_193, full_194]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_82, [x.reshape([]) for x in combine_64]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_81 = paddle._C_ops.matmul(reshape__128, parameter_226, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__100 = paddle._C_ops.add_(matmul_81, parameter_227)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__101 = paddle._C_ops.add_(add__96, add__100)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__101, parameter_228, parameter_229, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_82 = paddle._C_ops.matmul(layer_norm_114, parameter_230, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__102 = paddle._C_ops.add_(matmul_82, parameter_231)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_11 = paddle._C_ops.gelu(add__102, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_83 = paddle._C_ops.matmul(gelu_11, parameter_232, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__103 = paddle._C_ops.add_(matmul_83, parameter_233)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__104 = paddle._C_ops.add_(add__101, add__103)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__104, parameter_234, parameter_235, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(layer_norm_117, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_16, [0], full_int_array_98, full_int_array_99, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_84 = paddle._C_ops.matmul(layer_norm_117, parameter_236, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__105 = paddle._C_ops.add_(matmul_84, parameter_237)

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_40, full_195, full_196, full_197]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__105, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_83 = paddle._C_ops.transpose(reshape__130, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_84 = paddle._C_ops.transpose(layer_norm_117, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_198 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_199 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_200 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_40, full_198, full_199, full_200]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_84, [x.reshape([]) for x in combine_66]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_15 = paddle._C_ops.conv2d(reshape__132, parameter_238, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_100 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_239, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__106 = paddle._C_ops.add_(conv2d_15, reshape_36)

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_40, full_201, full_202]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__106, [x.reshape([]) for x in combine_67]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_85 = paddle._C_ops.transpose(reshape__134, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_85, parameter_240, parameter_241, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_85 = paddle._C_ops.matmul(layer_norm_120, parameter_242, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__107 = paddle._C_ops.add_(matmul_85, parameter_243)

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_68 = [slice_40, full_203, full_204, full_205, full_206]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__107, [x.reshape([]) for x in combine_68]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_86 = paddle._C_ops.transpose(reshape__136, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_86, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_86, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_87 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_86 = paddle._C_ops.matmul(transpose_83, transpose_87, False, False)

        # pd_op.full: (1xf32) <- ()
        full_207 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_86, full_207, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__12 = paddle._C_ops.softmax_(scale__12, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_87 = paddle._C_ops.matmul(softmax__12, slice_42, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_88 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_208 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_69 = [slice_40, full_208, full_209]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_88, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_88 = paddle._C_ops.matmul(reshape__138, parameter_244, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__108 = paddle._C_ops.add_(matmul_88, parameter_245)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__109 = paddle._C_ops.add_(add__104, add__108)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__109, parameter_246, parameter_247, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_89 = paddle._C_ops.matmul(layer_norm_123, parameter_248, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__110 = paddle._C_ops.add_(matmul_89, parameter_249)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_12 = paddle._C_ops.gelu(add__110, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_90 = paddle._C_ops.matmul(gelu_12, parameter_250, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__111 = paddle._C_ops.add_(matmul_90, parameter_251)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__112 = paddle._C_ops.add_(add__109, add__111)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__112, parameter_252, parameter_253, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(layer_norm_126, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(shape_17, [0], full_int_array_105, full_int_array_106, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_91 = paddle._C_ops.matmul(layer_norm_126, parameter_254, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__113 = paddle._C_ops.add_(matmul_91, parameter_255)

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_43, full_210, full_211, full_212]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__113, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_89 = paddle._C_ops.transpose(reshape__140, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_90 = paddle._C_ops.transpose(layer_norm_126, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_71 = [slice_43, full_213, full_214, full_215]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_90, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_16 = paddle._C_ops.conv2d(reshape__142, parameter_256, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_107 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_257, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__114 = paddle._C_ops.add_(conv2d_16, reshape_38)

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_72 = [slice_43, full_216, full_217]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__114, [x.reshape([]) for x in combine_72]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_91 = paddle._C_ops.transpose(reshape__144, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_91, parameter_258, parameter_259, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_92 = paddle._C_ops.matmul(layer_norm_129, parameter_260, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__115 = paddle._C_ops.add_(matmul_92, parameter_261)

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_221 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_43, full_218, full_219, full_220, full_221]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__115, [x.reshape([]) for x in combine_73]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_92 = paddle._C_ops.transpose(reshape__146, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(transpose_92, [0], full_int_array_108, full_int_array_109, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(transpose_92, [0], full_int_array_110, full_int_array_111, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_93 = paddle._C_ops.transpose(slice_44, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_93 = paddle._C_ops.matmul(transpose_89, transpose_93, False, False)

        # pd_op.full: (1xf32) <- ()
        full_222 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_93, full_222, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__13 = paddle._C_ops.softmax_(scale__13, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_94 = paddle._C_ops.matmul(softmax__13, slice_45, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_94 = paddle._C_ops.transpose(matmul_94, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_43, full_223, full_224]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_94, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_95 = paddle._C_ops.matmul(reshape__148, parameter_262, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__116 = paddle._C_ops.add_(matmul_95, parameter_263)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__117 = paddle._C_ops.add_(add__112, add__116)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__117, parameter_264, parameter_265, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_96 = paddle._C_ops.matmul(layer_norm_132, parameter_266, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__118 = paddle._C_ops.add_(matmul_96, parameter_267)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_13 = paddle._C_ops.gelu(add__118, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_97 = paddle._C_ops.matmul(gelu_13, parameter_268, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__119 = paddle._C_ops.add_(matmul_97, parameter_269)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__120 = paddle._C_ops.add_(add__117, add__119)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__120, parameter_270, parameter_271, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(layer_norm_135, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_18, [0], full_int_array_112, full_int_array_113, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_98 = paddle._C_ops.matmul(layer_norm_135, parameter_272, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__121 = paddle._C_ops.add_(matmul_98, parameter_273)

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_75 = [slice_46, full_225, full_226, full_227]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__121, [x.reshape([]) for x in combine_75]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_95 = paddle._C_ops.transpose(reshape__150, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_96 = paddle._C_ops.transpose(layer_norm_135, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_230 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_46, full_228, full_229, full_230]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_96, [x.reshape([]) for x in combine_76]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_17 = paddle._C_ops.conv2d(reshape__152, parameter_274, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_114 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_275, full_int_array_114), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__122 = paddle._C_ops.add_(conv2d_17, reshape_40)

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_77 = [slice_46, full_231, full_232]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__122, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_97 = paddle._C_ops.transpose(reshape__154, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_97, parameter_276, parameter_277, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_99 = paddle._C_ops.matmul(layer_norm_138, parameter_278, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__123 = paddle._C_ops.add_(matmul_99, parameter_279)

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_78 = [slice_46, full_233, full_234, full_235, full_236]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__123, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_98 = paddle._C_ops.transpose(reshape__156, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_98, [0], full_int_array_115, full_int_array_116, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_98, [0], full_int_array_117, full_int_array_118, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_99 = paddle._C_ops.transpose(slice_47, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_100 = paddle._C_ops.matmul(transpose_95, transpose_99, False, False)

        # pd_op.full: (1xf32) <- ()
        full_237 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_100, full_237, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__14 = paddle._C_ops.softmax_(scale__14, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_101 = paddle._C_ops.matmul(softmax__14, slice_48, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_100 = paddle._C_ops.transpose(matmul_101, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_79 = [slice_46, full_238, full_239]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_100, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_102 = paddle._C_ops.matmul(reshape__158, parameter_280, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__124 = paddle._C_ops.add_(matmul_102, parameter_281)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__125 = paddle._C_ops.add_(add__120, add__124)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__125, parameter_282, parameter_283, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_103 = paddle._C_ops.matmul(layer_norm_141, parameter_284, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__126 = paddle._C_ops.add_(matmul_103, parameter_285)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_14 = paddle._C_ops.gelu(add__126, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_104 = paddle._C_ops.matmul(gelu_14, parameter_286, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__127 = paddle._C_ops.add_(matmul_104, parameter_287)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__128 = paddle._C_ops.add_(add__125, add__127)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__128, parameter_288, parameter_289, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(layer_norm_144, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_19, [0], full_int_array_119, full_int_array_120, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_105 = paddle._C_ops.matmul(layer_norm_144, parameter_290, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__129 = paddle._C_ops.add_(matmul_105, parameter_291)

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_242 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_80 = [slice_49, full_240, full_241, full_242]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__129, [x.reshape([]) for x in combine_80]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_101 = paddle._C_ops.transpose(reshape__160, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_102 = paddle._C_ops.transpose(layer_norm_144, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_81 = [slice_49, full_243, full_244, full_245]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_102, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_18 = paddle._C_ops.conv2d(reshape__162, parameter_292, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_121 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_293, full_int_array_121), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__130 = paddle._C_ops.add_(conv2d_18, reshape_42)

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_82 = [slice_49, full_246, full_247]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__130, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_103 = paddle._C_ops.transpose(reshape__164, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_103, parameter_294, parameter_295, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_106 = paddle._C_ops.matmul(layer_norm_147, parameter_296, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__131 = paddle._C_ops.add_(matmul_106, parameter_297)

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_83 = [slice_49, full_248, full_249, full_250, full_251]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__131, [x.reshape([]) for x in combine_83]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_104 = paddle._C_ops.transpose(reshape__166, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_104, [0], full_int_array_122, full_int_array_123, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_104, [0], full_int_array_124, full_int_array_125, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_105 = paddle._C_ops.transpose(slice_50, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_107 = paddle._C_ops.matmul(transpose_101, transpose_105, False, False)

        # pd_op.full: (1xf32) <- ()
        full_252 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_107, full_252, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__15 = paddle._C_ops.softmax_(scale__15, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_108 = paddle._C_ops.matmul(softmax__15, slice_51, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_106 = paddle._C_ops.transpose(matmul_108, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_84 = [slice_49, full_253, full_254]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_106, [x.reshape([]) for x in combine_84]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_109 = paddle._C_ops.matmul(reshape__168, parameter_298, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__132 = paddle._C_ops.add_(matmul_109, parameter_299)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__133 = paddle._C_ops.add_(add__128, add__132)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__133, parameter_300, parameter_301, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_110 = paddle._C_ops.matmul(layer_norm_150, parameter_302, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__134 = paddle._C_ops.add_(matmul_110, parameter_303)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_15 = paddle._C_ops.gelu(add__134, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_111 = paddle._C_ops.matmul(gelu_15, parameter_304, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__135 = paddle._C_ops.add_(matmul_111, parameter_305)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__136 = paddle._C_ops.add_(add__133, add__135)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__136, parameter_306, parameter_307, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(layer_norm_153, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_20, [0], full_int_array_126, full_int_array_127, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_112 = paddle._C_ops.matmul(layer_norm_153, parameter_308, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__137 = paddle._C_ops.add_(matmul_112, parameter_309)

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_85 = [slice_52, full_255, full_256, full_257]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__137, [x.reshape([]) for x in combine_85]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_107 = paddle._C_ops.transpose(reshape__170, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_108 = paddle._C_ops.transpose(layer_norm_153, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_260 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_86 = [slice_52, full_258, full_259, full_260]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_108, [x.reshape([]) for x in combine_86]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_19 = paddle._C_ops.conv2d(reshape__172, parameter_310, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_128 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_311, full_int_array_128), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__138 = paddle._C_ops.add_(conv2d_19, reshape_44)

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_87 = [slice_52, full_261, full_262]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__138, [x.reshape([]) for x in combine_87]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_109 = paddle._C_ops.transpose(reshape__174, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_109, parameter_312, parameter_313, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_113 = paddle._C_ops.matmul(layer_norm_156, parameter_314, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__139 = paddle._C_ops.add_(matmul_113, parameter_315)

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_264 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_88 = [slice_52, full_263, full_264, full_265, full_266]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__139, [x.reshape([]) for x in combine_88]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_110 = paddle._C_ops.transpose(reshape__176, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(transpose_110, [0], full_int_array_129, full_int_array_130, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(transpose_110, [0], full_int_array_131, full_int_array_132, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_111 = paddle._C_ops.transpose(slice_53, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_114 = paddle._C_ops.matmul(transpose_107, transpose_111, False, False)

        # pd_op.full: (1xf32) <- ()
        full_267 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_114, full_267, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__16 = paddle._C_ops.softmax_(scale__16, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_115 = paddle._C_ops.matmul(softmax__16, slice_54, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_112 = paddle._C_ops.transpose(matmul_115, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_268 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_269 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_89 = [slice_52, full_268, full_269]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_112, [x.reshape([]) for x in combine_89]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_116 = paddle._C_ops.matmul(reshape__178, parameter_316, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__140 = paddle._C_ops.add_(matmul_116, parameter_317)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__141 = paddle._C_ops.add_(add__136, add__140)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_159, layer_norm_160, layer_norm_161 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__141, parameter_318, parameter_319, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_117 = paddle._C_ops.matmul(layer_norm_159, parameter_320, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__142 = paddle._C_ops.add_(matmul_117, parameter_321)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_16 = paddle._C_ops.gelu(add__142, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_118 = paddle._C_ops.matmul(gelu_16, parameter_322, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__143 = paddle._C_ops.add_(matmul_118, parameter_323)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__144 = paddle._C_ops.add_(add__141, add__143)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_162, layer_norm_163, layer_norm_164 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__144, parameter_324, parameter_325, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(layer_norm_162, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(shape_21, [0], full_int_array_133, full_int_array_134, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_119 = paddle._C_ops.matmul(layer_norm_162, parameter_326, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__145 = paddle._C_ops.add_(matmul_119, parameter_327)

        # pd_op.full: (1xi32) <- ()
        full_270 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_271 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_272 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_90 = [slice_55, full_270, full_271, full_272]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__145, [x.reshape([]) for x in combine_90]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_113 = paddle._C_ops.transpose(reshape__180, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_114 = paddle._C_ops.transpose(layer_norm_162, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_273 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_274 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_275 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_91 = [slice_55, full_273, full_274, full_275]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_114, [x.reshape([]) for x in combine_91]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_20 = paddle._C_ops.conv2d(reshape__182, parameter_328, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_135 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_329, full_int_array_135), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__146 = paddle._C_ops.add_(conv2d_20, reshape_46)

        # pd_op.full: (1xi32) <- ()
        full_276 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_277 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_92 = [slice_55, full_276, full_277]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__146, [x.reshape([]) for x in combine_92]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_115 = paddle._C_ops.transpose(reshape__184, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_165, layer_norm_166, layer_norm_167 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_115, parameter_330, parameter_331, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_120 = paddle._C_ops.matmul(layer_norm_165, parameter_332, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__147 = paddle._C_ops.add_(matmul_120, parameter_333)

        # pd_op.full: (1xi32) <- ()
        full_278 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_279 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_280 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_281 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_93 = [slice_55, full_278, full_279, full_280, full_281]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__147, [x.reshape([]) for x in combine_93]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_116 = paddle._C_ops.transpose(reshape__186, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(transpose_116, [0], full_int_array_136, full_int_array_137, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(transpose_116, [0], full_int_array_138, full_int_array_139, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_117 = paddle._C_ops.transpose(slice_56, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_121 = paddle._C_ops.matmul(transpose_113, transpose_117, False, False)

        # pd_op.full: (1xf32) <- ()
        full_282 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_121, full_282, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__17 = paddle._C_ops.softmax_(scale__17, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_122 = paddle._C_ops.matmul(softmax__17, slice_57, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_118 = paddle._C_ops.transpose(matmul_122, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_283 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_284 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_94 = [slice_55, full_283, full_284]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__188, reshape__189 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_118, [x.reshape([]) for x in combine_94]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_123 = paddle._C_ops.matmul(reshape__188, parameter_334, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__148 = paddle._C_ops.add_(matmul_123, parameter_335)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__149 = paddle._C_ops.add_(add__144, add__148)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_168, layer_norm_169, layer_norm_170 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__149, parameter_336, parameter_337, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_124 = paddle._C_ops.matmul(layer_norm_168, parameter_338, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__150 = paddle._C_ops.add_(matmul_124, parameter_339)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_17 = paddle._C_ops.gelu(add__150, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_125 = paddle._C_ops.matmul(gelu_17, parameter_340, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__151 = paddle._C_ops.add_(matmul_125, parameter_341)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__152 = paddle._C_ops.add_(add__149, add__151)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_171, layer_norm_172, layer_norm_173 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__152, parameter_342, parameter_343, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(layer_norm_171, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(shape_22, [0], full_int_array_140, full_int_array_141, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_126 = paddle._C_ops.matmul(layer_norm_171, parameter_344, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__153 = paddle._C_ops.add_(matmul_126, parameter_345)

        # pd_op.full: (1xi32) <- ()
        full_285 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_286 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_287 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_95 = [slice_58, full_285, full_286, full_287]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__190, reshape__191 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__153, [x.reshape([]) for x in combine_95]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_119 = paddle._C_ops.transpose(reshape__190, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_120 = paddle._C_ops.transpose(layer_norm_171, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_288 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_289 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_290 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_96 = [slice_58, full_288, full_289, full_290]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__192, reshape__193 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_120, [x.reshape([]) for x in combine_96]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_21 = paddle._C_ops.conv2d(reshape__192, parameter_346, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_142 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_347, full_int_array_142), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__154 = paddle._C_ops.add_(conv2d_21, reshape_48)

        # pd_op.full: (1xi32) <- ()
        full_291 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_292 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_97 = [slice_58, full_291, full_292]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__194, reshape__195 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__154, [x.reshape([]) for x in combine_97]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_121 = paddle._C_ops.transpose(reshape__194, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_174, layer_norm_175, layer_norm_176 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_121, parameter_348, parameter_349, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_127 = paddle._C_ops.matmul(layer_norm_174, parameter_350, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__155 = paddle._C_ops.add_(matmul_127, parameter_351)

        # pd_op.full: (1xi32) <- ()
        full_293 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_294 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_295 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_296 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_98 = [slice_58, full_293, full_294, full_295, full_296]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__196, reshape__197 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__155, [x.reshape([]) for x in combine_98]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_122 = paddle._C_ops.transpose(reshape__196, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_122, [0], full_int_array_143, full_int_array_144, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(transpose_122, [0], full_int_array_145, full_int_array_146, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_123 = paddle._C_ops.transpose(slice_59, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_128 = paddle._C_ops.matmul(transpose_119, transpose_123, False, False)

        # pd_op.full: (1xf32) <- ()
        full_297 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_128, full_297, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__18 = paddle._C_ops.softmax_(scale__18, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_129 = paddle._C_ops.matmul(softmax__18, slice_60, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_124 = paddle._C_ops.transpose(matmul_129, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_298 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_299 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_99 = [slice_58, full_298, full_299]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__198, reshape__199 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_124, [x.reshape([]) for x in combine_99]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_130 = paddle._C_ops.matmul(reshape__198, parameter_352, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__156 = paddle._C_ops.add_(matmul_130, parameter_353)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__157 = paddle._C_ops.add_(add__152, add__156)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_177, layer_norm_178, layer_norm_179 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__157, parameter_354, parameter_355, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_131 = paddle._C_ops.matmul(layer_norm_177, parameter_356, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__158 = paddle._C_ops.add_(matmul_131, parameter_357)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_18 = paddle._C_ops.gelu(add__158, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_132 = paddle._C_ops.matmul(gelu_18, parameter_358, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__159 = paddle._C_ops.add_(matmul_132, parameter_359)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__160 = paddle._C_ops.add_(add__157, add__159)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_180, layer_norm_181, layer_norm_182 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__160, parameter_360, parameter_361, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(layer_norm_180, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(shape_23, [0], full_int_array_147, full_int_array_148, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_133 = paddle._C_ops.matmul(layer_norm_180, parameter_362, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__161 = paddle._C_ops.add_(matmul_133, parameter_363)

        # pd_op.full: (1xi32) <- ()
        full_300 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_301 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_302 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_100 = [slice_61, full_300, full_301, full_302]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__200, reshape__201 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__161, [x.reshape([]) for x in combine_100]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_125 = paddle._C_ops.transpose(reshape__200, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_126 = paddle._C_ops.transpose(layer_norm_180, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_303 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_304 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_305 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_101 = [slice_61, full_303, full_304, full_305]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__202, reshape__203 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_126, [x.reshape([]) for x in combine_101]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_22 = paddle._C_ops.conv2d(reshape__202, parameter_364, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_149 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_365, full_int_array_149), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__162 = paddle._C_ops.add_(conv2d_22, reshape_50)

        # pd_op.full: (1xi32) <- ()
        full_306 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_307 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_102 = [slice_61, full_306, full_307]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__204, reshape__205 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__162, [x.reshape([]) for x in combine_102]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_127 = paddle._C_ops.transpose(reshape__204, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_183, layer_norm_184, layer_norm_185 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_127, parameter_366, parameter_367, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_134 = paddle._C_ops.matmul(layer_norm_183, parameter_368, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__163 = paddle._C_ops.add_(matmul_134, parameter_369)

        # pd_op.full: (1xi32) <- ()
        full_308 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_309 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_310 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_311 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_103 = [slice_61, full_308, full_309, full_310, full_311]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__206, reshape__207 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__163, [x.reshape([]) for x in combine_103]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_128 = paddle._C_ops.transpose(reshape__206, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(transpose_128, [0], full_int_array_150, full_int_array_151, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(transpose_128, [0], full_int_array_152, full_int_array_153, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_129 = paddle._C_ops.transpose(slice_62, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_135 = paddle._C_ops.matmul(transpose_125, transpose_129, False, False)

        # pd_op.full: (1xf32) <- ()
        full_312 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_135, full_312, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__19 = paddle._C_ops.softmax_(scale__19, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_136 = paddle._C_ops.matmul(softmax__19, slice_63, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_130 = paddle._C_ops.transpose(matmul_136, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_313 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_314 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_104 = [slice_61, full_313, full_314]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__208, reshape__209 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_130, [x.reshape([]) for x in combine_104]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_137 = paddle._C_ops.matmul(reshape__208, parameter_370, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__164 = paddle._C_ops.add_(matmul_137, parameter_371)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__165 = paddle._C_ops.add_(add__160, add__164)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_186, layer_norm_187, layer_norm_188 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__165, parameter_372, parameter_373, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_138 = paddle._C_ops.matmul(layer_norm_186, parameter_374, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__166 = paddle._C_ops.add_(matmul_138, parameter_375)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_19 = paddle._C_ops.gelu(add__166, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_139 = paddle._C_ops.matmul(gelu_19, parameter_376, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__167 = paddle._C_ops.add_(matmul_139, parameter_377)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__168 = paddle._C_ops.add_(add__165, add__167)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_189, layer_norm_190, layer_norm_191 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__168, parameter_378, parameter_379, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(layer_norm_189, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_24, [0], full_int_array_154, full_int_array_155, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_140 = paddle._C_ops.matmul(layer_norm_189, parameter_380, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__169 = paddle._C_ops.add_(matmul_140, parameter_381)

        # pd_op.full: (1xi32) <- ()
        full_315 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_316 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_317 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_105 = [slice_64, full_315, full_316, full_317]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__210, reshape__211 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__169, [x.reshape([]) for x in combine_105]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_131 = paddle._C_ops.transpose(reshape__210, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_132 = paddle._C_ops.transpose(layer_norm_189, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_318 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_319 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_320 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_106 = [slice_64, full_318, full_319, full_320]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__212, reshape__213 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_132, [x.reshape([]) for x in combine_106]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_23 = paddle._C_ops.conv2d(reshape__212, parameter_382, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_156 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_383, full_int_array_156), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__170 = paddle._C_ops.add_(conv2d_23, reshape_52)

        # pd_op.full: (1xi32) <- ()
        full_321 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_322 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_107 = [slice_64, full_321, full_322]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__214, reshape__215 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__170, [x.reshape([]) for x in combine_107]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_133 = paddle._C_ops.transpose(reshape__214, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_192, layer_norm_193, layer_norm_194 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_133, parameter_384, parameter_385, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_141 = paddle._C_ops.matmul(layer_norm_192, parameter_386, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__171 = paddle._C_ops.add_(matmul_141, parameter_387)

        # pd_op.full: (1xi32) <- ()
        full_323 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_324 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_325 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_326 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_108 = [slice_64, full_323, full_324, full_325, full_326]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__216, reshape__217 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__171, [x.reshape([]) for x in combine_108]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_134 = paddle._C_ops.transpose(reshape__216, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(transpose_134, [0], full_int_array_157, full_int_array_158, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_134, [0], full_int_array_159, full_int_array_160, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_135 = paddle._C_ops.transpose(slice_65, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_142 = paddle._C_ops.matmul(transpose_131, transpose_135, False, False)

        # pd_op.full: (1xf32) <- ()
        full_327 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_142, full_327, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__20 = paddle._C_ops.softmax_(scale__20, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_143 = paddle._C_ops.matmul(softmax__20, slice_66, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_136 = paddle._C_ops.transpose(matmul_143, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_328 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_329 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_109 = [slice_64, full_328, full_329]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__218, reshape__219 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_136, [x.reshape([]) for x in combine_109]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_144 = paddle._C_ops.matmul(reshape__218, parameter_388, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__172 = paddle._C_ops.add_(matmul_144, parameter_389)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__173 = paddle._C_ops.add_(add__168, add__172)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_195, layer_norm_196, layer_norm_197 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__173, parameter_390, parameter_391, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_145 = paddle._C_ops.matmul(layer_norm_195, parameter_392, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__174 = paddle._C_ops.add_(matmul_145, parameter_393)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_20 = paddle._C_ops.gelu(add__174, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_146 = paddle._C_ops.matmul(gelu_20, parameter_394, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__175 = paddle._C_ops.add_(matmul_146, parameter_395)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__176 = paddle._C_ops.add_(add__173, add__175)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_198, layer_norm_199, layer_norm_200 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__176, parameter_396, parameter_397, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(layer_norm_198, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(shape_25, [0], full_int_array_161, full_int_array_162, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_147 = paddle._C_ops.matmul(layer_norm_198, parameter_398, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__177 = paddle._C_ops.add_(matmul_147, parameter_399)

        # pd_op.full: (1xi32) <- ()
        full_330 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_331 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_332 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_110 = [slice_67, full_330, full_331, full_332]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__220, reshape__221 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__177, [x.reshape([]) for x in combine_110]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_137 = paddle._C_ops.transpose(reshape__220, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_138 = paddle._C_ops.transpose(layer_norm_198, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_333 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_334 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_335 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_111 = [slice_67, full_333, full_334, full_335]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__222, reshape__223 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_138, [x.reshape([]) for x in combine_111]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_24 = paddle._C_ops.conv2d(reshape__222, parameter_400, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_163 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_401, full_int_array_163), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__178 = paddle._C_ops.add_(conv2d_24, reshape_54)

        # pd_op.full: (1xi32) <- ()
        full_336 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_337 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_112 = [slice_67, full_336, full_337]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__224, reshape__225 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__178, [x.reshape([]) for x in combine_112]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_139 = paddle._C_ops.transpose(reshape__224, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_201, layer_norm_202, layer_norm_203 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_139, parameter_402, parameter_403, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_148 = paddle._C_ops.matmul(layer_norm_201, parameter_404, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__179 = paddle._C_ops.add_(matmul_148, parameter_405)

        # pd_op.full: (1xi32) <- ()
        full_338 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_339 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_340 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_341 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_113 = [slice_67, full_338, full_339, full_340, full_341]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__226, reshape__227 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__179, [x.reshape([]) for x in combine_113]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_140 = paddle._C_ops.transpose(reshape__226, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(transpose_140, [0], full_int_array_164, full_int_array_165, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(transpose_140, [0], full_int_array_166, full_int_array_167, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_141 = paddle._C_ops.transpose(slice_68, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_149 = paddle._C_ops.matmul(transpose_137, transpose_141, False, False)

        # pd_op.full: (1xf32) <- ()
        full_342 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_149, full_342, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__21 = paddle._C_ops.softmax_(scale__21, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_150 = paddle._C_ops.matmul(softmax__21, slice_69, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_142 = paddle._C_ops.transpose(matmul_150, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_343 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_344 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_114 = [slice_67, full_343, full_344]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__228, reshape__229 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_142, [x.reshape([]) for x in combine_114]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_151 = paddle._C_ops.matmul(reshape__228, parameter_406, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__180 = paddle._C_ops.add_(matmul_151, parameter_407)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__181 = paddle._C_ops.add_(add__176, add__180)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_204, layer_norm_205, layer_norm_206 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__181, parameter_408, parameter_409, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_152 = paddle._C_ops.matmul(layer_norm_204, parameter_410, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__182 = paddle._C_ops.add_(matmul_152, parameter_411)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_21 = paddle._C_ops.gelu(add__182, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_153 = paddle._C_ops.matmul(gelu_21, parameter_412, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__183 = paddle._C_ops.add_(matmul_153, parameter_413)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__184 = paddle._C_ops.add_(add__181, add__183)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_207, layer_norm_208, layer_norm_209 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__184, parameter_414, parameter_415, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(layer_norm_207, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_26, [0], full_int_array_168, full_int_array_169, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_154 = paddle._C_ops.matmul(layer_norm_207, parameter_416, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__185 = paddle._C_ops.add_(matmul_154, parameter_417)

        # pd_op.full: (1xi32) <- ()
        full_345 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_346 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_347 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_115 = [slice_70, full_345, full_346, full_347]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__230, reshape__231 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__185, [x.reshape([]) for x in combine_115]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_143 = paddle._C_ops.transpose(reshape__230, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_144 = paddle._C_ops.transpose(layer_norm_207, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_348 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_349 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_350 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_116 = [slice_70, full_348, full_349, full_350]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__232, reshape__233 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_144, [x.reshape([]) for x in combine_116]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_25 = paddle._C_ops.conv2d(reshape__232, parameter_418, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_170 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_419, full_int_array_170), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__186 = paddle._C_ops.add_(conv2d_25, reshape_56)

        # pd_op.full: (1xi32) <- ()
        full_351 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_352 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_117 = [slice_70, full_351, full_352]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__234, reshape__235 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__186, [x.reshape([]) for x in combine_117]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_145 = paddle._C_ops.transpose(reshape__234, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_210, layer_norm_211, layer_norm_212 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_145, parameter_420, parameter_421, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_155 = paddle._C_ops.matmul(layer_norm_210, parameter_422, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__187 = paddle._C_ops.add_(matmul_155, parameter_423)

        # pd_op.full: (1xi32) <- ()
        full_353 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_354 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_355 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_356 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_118 = [slice_70, full_353, full_354, full_355, full_356]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__236, reshape__237 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__187, [x.reshape([]) for x in combine_118]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_146 = paddle._C_ops.transpose(reshape__236, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(transpose_146, [0], full_int_array_171, full_int_array_172, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_146, [0], full_int_array_173, full_int_array_174, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_147 = paddle._C_ops.transpose(slice_71, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_156 = paddle._C_ops.matmul(transpose_143, transpose_147, False, False)

        # pd_op.full: (1xf32) <- ()
        full_357 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_156, full_357, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__22 = paddle._C_ops.softmax_(scale__22, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_157 = paddle._C_ops.matmul(softmax__22, slice_72, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_148 = paddle._C_ops.transpose(matmul_157, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_358 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_359 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_119 = [slice_70, full_358, full_359]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__238, reshape__239 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_148, [x.reshape([]) for x in combine_119]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_158 = paddle._C_ops.matmul(reshape__238, parameter_424, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__188 = paddle._C_ops.add_(matmul_158, parameter_425)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__189 = paddle._C_ops.add_(add__184, add__188)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_213, layer_norm_214, layer_norm_215 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__189, parameter_426, parameter_427, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_159 = paddle._C_ops.matmul(layer_norm_213, parameter_428, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__190 = paddle._C_ops.add_(matmul_159, parameter_429)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_22 = paddle._C_ops.gelu(add__190, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_160 = paddle._C_ops.matmul(gelu_22, parameter_430, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__191 = paddle._C_ops.add_(matmul_160, parameter_431)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__192 = paddle._C_ops.add_(add__189, add__191)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_216, layer_norm_217, layer_norm_218 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__192, parameter_432, parameter_433, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(layer_norm_216, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(shape_27, [0], full_int_array_175, full_int_array_176, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_161 = paddle._C_ops.matmul(layer_norm_216, parameter_434, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__193 = paddle._C_ops.add_(matmul_161, parameter_435)

        # pd_op.full: (1xi32) <- ()
        full_360 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_361 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_362 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_120 = [slice_73, full_360, full_361, full_362]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__240, reshape__241 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__193, [x.reshape([]) for x in combine_120]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_149 = paddle._C_ops.transpose(reshape__240, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_150 = paddle._C_ops.transpose(layer_norm_216, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_363 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_364 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_365 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_121 = [slice_73, full_363, full_364, full_365]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__242, reshape__243 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_150, [x.reshape([]) for x in combine_121]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_26 = paddle._C_ops.conv2d(reshape__242, parameter_436, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_177 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_437, full_int_array_177), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__194 = paddle._C_ops.add_(conv2d_26, reshape_58)

        # pd_op.full: (1xi32) <- ()
        full_366 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_367 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_122 = [slice_73, full_366, full_367]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__244, reshape__245 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__194, [x.reshape([]) for x in combine_122]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_151 = paddle._C_ops.transpose(reshape__244, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_219, layer_norm_220, layer_norm_221 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_151, parameter_438, parameter_439, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_162 = paddle._C_ops.matmul(layer_norm_219, parameter_440, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__195 = paddle._C_ops.add_(matmul_162, parameter_441)

        # pd_op.full: (1xi32) <- ()
        full_368 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_369 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_370 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_371 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_123 = [slice_73, full_368, full_369, full_370, full_371]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__246, reshape__247 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__195, [x.reshape([]) for x in combine_123]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_152 = paddle._C_ops.transpose(reshape__246, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(transpose_152, [0], full_int_array_178, full_int_array_179, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_152, [0], full_int_array_180, full_int_array_181, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_153 = paddle._C_ops.transpose(slice_74, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_163 = paddle._C_ops.matmul(transpose_149, transpose_153, False, False)

        # pd_op.full: (1xf32) <- ()
        full_372 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_163, full_372, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__23 = paddle._C_ops.softmax_(scale__23, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_164 = paddle._C_ops.matmul(softmax__23, slice_75, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_154 = paddle._C_ops.transpose(matmul_164, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_373 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_374 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_124 = [slice_73, full_373, full_374]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__248, reshape__249 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_154, [x.reshape([]) for x in combine_124]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_165 = paddle._C_ops.matmul(reshape__248, parameter_442, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__196 = paddle._C_ops.add_(matmul_165, parameter_443)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__197 = paddle._C_ops.add_(add__192, add__196)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_222, layer_norm_223, layer_norm_224 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__197, parameter_444, parameter_445, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_166 = paddle._C_ops.matmul(layer_norm_222, parameter_446, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__198 = paddle._C_ops.add_(matmul_166, parameter_447)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_23 = paddle._C_ops.gelu(add__198, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_167 = paddle._C_ops.matmul(gelu_23, parameter_448, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__199 = paddle._C_ops.add_(matmul_167, parameter_449)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__200 = paddle._C_ops.add_(add__197, add__199)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_225, layer_norm_226, layer_norm_227 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__200, parameter_450, parameter_451, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x196x320xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(layer_norm_225, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_28, [0], full_int_array_182, full_int_array_183, [1], [0])

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_168 = paddle._C_ops.matmul(layer_norm_225, parameter_452, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__201 = paddle._C_ops.add_(matmul_168, parameter_453)

        # pd_op.full: (1xi32) <- ()
        full_375 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_376 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_377 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_125 = [slice_76, full_375, full_376, full_377]

        # pd_op.reshape_: (-1x196x5x64xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__250, reshape__251 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__201, [x.reshape([]) for x in combine_125]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x5x196x64xf16) <- (-1x196x5x64xf16)
        transpose_155 = paddle._C_ops.transpose(reshape__250, [0, 2, 1, 3])

        # pd_op.transpose: (-1x320x196xf16) <- (-1x196x320xf16)
        transpose_156 = paddle._C_ops.transpose(layer_norm_225, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_378 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_379 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_380 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_126 = [slice_76, full_378, full_379, full_380]

        # pd_op.reshape_: (-1x320x14x14xf16, 0x-1x320x196xf16) <- (-1x320x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__252, reshape__253 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_156, [x.reshape([]) for x in combine_126]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x320x7x7xf16) <- (-1x320x14x14xf16, 320x320x2x2xf16)
        conv2d_27 = paddle._C_ops.conv2d(reshape__252, parameter_454, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_184 = [1, 320, 1, 1]

        # pd_op.reshape: (1x320x1x1xf16, 0x320xf16) <- (320xf16, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_455, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x320x7x7xf16) <- (-1x320x7x7xf16, 1x320x1x1xf16)
        add__202 = paddle._C_ops.add_(conv2d_27, reshape_60)

        # pd_op.full: (1xi32) <- ()
        full_381 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_382 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_127 = [slice_76, full_381, full_382]

        # pd_op.reshape_: (-1x320x49xf16, 0x-1x320x7x7xf16) <- (-1x320x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__254, reshape__255 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__202, [x.reshape([]) for x in combine_127]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x320xf16) <- (-1x320x49xf16)
        transpose_157 = paddle._C_ops.transpose(reshape__254, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x320xf16, -49xf32, -49xf32) <- (-1x49x320xf16, 320xf32, 320xf32)
        layer_norm_228, layer_norm_229, layer_norm_230 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_157, parameter_456, parameter_457, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x640xf16) <- (-1x49x320xf16, 320x640xf16)
        matmul_169 = paddle._C_ops.matmul(layer_norm_228, parameter_458, False, False)

        # pd_op.add_: (-1x49x640xf16) <- (-1x49x640xf16, 640xf16)
        add__203 = paddle._C_ops.add_(matmul_169, parameter_459)

        # pd_op.full: (1xi32) <- ()
        full_383 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_384 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_385 = paddle._C_ops.full([1], float('5'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_386 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_128 = [slice_76, full_383, full_384, full_385, full_386]

        # pd_op.reshape_: (-1x49x2x5x64xf16, 0x-1x49x640xf16) <- (-1x49x640xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__256, reshape__257 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__203, [x.reshape([]) for x in combine_128]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x5x49x64xf16) <- (-1x49x2x5x64xf16)
        transpose_158 = paddle._C_ops.transpose(reshape__256, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [1]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(transpose_158, [0], full_int_array_185, full_int_array_186, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [2]

        # pd_op.slice: (-1x5x49x64xf16) <- (2x-1x5x49x64xf16, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(transpose_158, [0], full_int_array_187, full_int_array_188, [1], [0])

        # pd_op.transpose: (-1x5x64x49xf16) <- (-1x5x49x64xf16)
        transpose_159 = paddle._C_ops.transpose(slice_77, [0, 1, 3, 2])

        # pd_op.matmul: (-1x5x196x49xf16) <- (-1x5x196x64xf16, -1x5x64x49xf16)
        matmul_170 = paddle._C_ops.matmul(transpose_155, transpose_159, False, False)

        # pd_op.full: (1xf32) <- ()
        full_387 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x5x196x49xf16) <- (-1x5x196x49xf16, 1xf32)
        scale__24 = paddle._C_ops.scale_(matmul_170, full_387, float('0'), True)

        # pd_op.softmax_: (-1x5x196x49xf16) <- (-1x5x196x49xf16)
        softmax__24 = paddle._C_ops.softmax_(scale__24, -1)

        # pd_op.matmul: (-1x5x196x64xf16) <- (-1x5x196x49xf16, -1x5x49x64xf16)
        matmul_171 = paddle._C_ops.matmul(softmax__24, slice_78, False, False)

        # pd_op.transpose: (-1x196x5x64xf16) <- (-1x5x196x64xf16)
        transpose_160 = paddle._C_ops.transpose(matmul_171, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_388 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_389 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_129 = [slice_76, full_388, full_389]

        # pd_op.reshape_: (-1x196x320xf16, 0x-1x196x5x64xf16) <- (-1x196x5x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__258, reshape__259 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_160, [x.reshape([]) for x in combine_129]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x320xf16, 320x320xf16)
        matmul_172 = paddle._C_ops.matmul(reshape__258, parameter_460, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__204 = paddle._C_ops.add_(matmul_172, parameter_461)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__205 = paddle._C_ops.add_(add__200, add__204)

        # pd_op.layer_norm: (-1x196x320xf16, -196xf32, -196xf32) <- (-1x196x320xf16, 320xf32, 320xf32)
        layer_norm_231, layer_norm_232, layer_norm_233 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__205, parameter_462, parameter_463, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x1280xf16) <- (-1x196x320xf16, 320x1280xf16)
        matmul_173 = paddle._C_ops.matmul(layer_norm_231, parameter_464, False, False)

        # pd_op.add_: (-1x196x1280xf16) <- (-1x196x1280xf16, 1280xf16)
        add__206 = paddle._C_ops.add_(matmul_173, parameter_465)

        # pd_op.gelu: (-1x196x1280xf16) <- (-1x196x1280xf16)
        gelu_24 = paddle._C_ops.gelu(add__206, False)

        # pd_op.matmul: (-1x196x320xf16) <- (-1x196x1280xf16, 1280x320xf16)
        matmul_174 = paddle._C_ops.matmul(gelu_24, parameter_466, False, False)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, 320xf16)
        add__207 = paddle._C_ops.add_(matmul_174, parameter_467)

        # pd_op.add_: (-1x196x320xf16) <- (-1x196x320xf16, -1x196x320xf16)
        add__208 = paddle._C_ops.add_(add__205, add__207)

        # pd_op.full: (1xi32) <- ()
        full_390 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_391 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_392 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_130 = [slice_0, full_390, full_391, full_392]

        # pd_op.reshape_: (-1x14x14x320xf16, 0x-1x196x320xf16) <- (-1x196x320xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__260, reshape__261 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__208, [x.reshape([]) for x in combine_130]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x320x14x14xf16) <- (-1x14x14x320xf16)
        transpose_161 = paddle._C_ops.transpose(reshape__260, [0, 3, 1, 2])

        # pd_op.conv2d: (-1x512x7x7xf16) <- (-1x320x14x14xf16, 512x320x2x2xf16)
        conv2d_28 = paddle._C_ops.conv2d(transpose_161, parameter_468, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_189 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_469, full_int_array_189), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__209 = paddle._C_ops.add_(conv2d_28, reshape_62)

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__12, flatten__13 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__209, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_162 = paddle._C_ops.transpose(flatten__12, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_234, layer_norm_235, layer_norm_236 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_162, parameter_470, parameter_471, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_237, layer_norm_238, layer_norm_239 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_234, parameter_472, parameter_473, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_29 = paddle._C_ops.shape(paddle.cast(layer_norm_237, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(shape_29, [0], full_int_array_190, full_int_array_191, [1], [0])

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_175 = paddle._C_ops.matmul(layer_norm_237, parameter_474, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__210 = paddle._C_ops.add_(matmul_175, parameter_475)

        # pd_op.full: (1xi32) <- ()
        full_393 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_394 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_395 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_131 = [slice_79, full_393, full_394, full_395]

        # pd_op.reshape_: (-1x49x8x64xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__262, reshape__263 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__210, [x.reshape([]) for x in combine_131]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x64xf16) <- (-1x49x8x64xf16)
        transpose_163 = paddle._C_ops.transpose(reshape__262, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x1024xf16) <- (-1x49x512xf16, 512x1024xf16)
        matmul_176 = paddle._C_ops.matmul(layer_norm_237, parameter_476, False, False)

        # pd_op.add_: (-1x49x1024xf16) <- (-1x49x1024xf16, 1024xf16)
        add__211 = paddle._C_ops.add_(matmul_176, parameter_477)

        # pd_op.full: (1xi32) <- ()
        full_396 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_397 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_398 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_399 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_132 = [slice_79, full_396, full_397, full_398, full_399]

        # pd_op.reshape_: (-1x49x2x8x64xf16, 0x-1x49x1024xf16) <- (-1x49x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__264, reshape__265 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__211, [x.reshape([]) for x in combine_132]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x64xf16) <- (-1x49x2x8x64xf16)
        transpose_164 = paddle._C_ops.transpose(reshape__264, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [1]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_164, [0], full_int_array_192, full_int_array_193, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [2]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(transpose_164, [0], full_int_array_194, full_int_array_195, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_165 = paddle._C_ops.transpose(slice_80, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_177 = paddle._C_ops.matmul(transpose_163, transpose_165, False, False)

        # pd_op.full: (1xf32) <- ()
        full_400 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__25 = paddle._C_ops.scale_(matmul_177, full_400, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__25 = paddle._C_ops.softmax_(scale__25, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_178 = paddle._C_ops.matmul(softmax__25, slice_81, False, False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_166 = paddle._C_ops.transpose(matmul_178, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_401 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_402 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_133 = [slice_79, full_401, full_402]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__266, reshape__267 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_166, [x.reshape([]) for x in combine_133]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_179 = paddle._C_ops.matmul(reshape__266, parameter_478, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__212 = paddle._C_ops.add_(matmul_179, parameter_479)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__213 = paddle._C_ops.add_(layer_norm_234, add__212)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_240, layer_norm_241, layer_norm_242 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__213, parameter_480, parameter_481, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_180 = paddle._C_ops.matmul(layer_norm_240, parameter_482, False, False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__214 = paddle._C_ops.add_(matmul_180, parameter_483)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_25 = paddle._C_ops.gelu(add__214, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_181 = paddle._C_ops.matmul(gelu_25, parameter_484, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__215 = paddle._C_ops.add_(matmul_181, parameter_485)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__216 = paddle._C_ops.add_(add__213, add__215)

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_30 = paddle._C_ops.shape(paddle.cast(add__216, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(shape_30, [0], full_int_array_196, full_int_array_197, [1], [0])

        # pd_op.transpose: (-1x512x49xf16) <- (-1x49x512xf16)
        transpose_167 = paddle._C_ops.transpose(add__216, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_403 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_404 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_405 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_134 = [slice_82, full_403, full_404, full_405]

        # pd_op.reshape_: (-1x512x7x7xf16, 0x-1x512x49xf16) <- (-1x512x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__268, reshape__269 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_167, [x.reshape([]) for x in combine_134]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 512x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__268, parameter_486, [1, 1], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_198 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_487, full_int_array_198), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__217 = paddle._C_ops.add_(depthwise_conv2d_3, reshape_64)

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__218 = paddle._C_ops.add_(add__217, reshape__268)

        # pd_op.flatten_: (-1x512x49xf16, None) <- (-1x512x7x7xf16)
        flatten__14, flatten__15 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__218, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x512xf16) <- (-1x512x49xf16)
        transpose_168 = paddle._C_ops.transpose(flatten__14, [0, 2, 1])

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_243, layer_norm_244, layer_norm_245 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_168, parameter_488, parameter_489, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_31 = paddle._C_ops.shape(paddle.cast(layer_norm_243, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(shape_31, [0], full_int_array_199, full_int_array_200, [1], [0])

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_182 = paddle._C_ops.matmul(layer_norm_243, parameter_490, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__219 = paddle._C_ops.add_(matmul_182, parameter_491)

        # pd_op.full: (1xi32) <- ()
        full_406 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_407 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_408 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_135 = [slice_83, full_406, full_407, full_408]

        # pd_op.reshape_: (-1x49x8x64xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__270, reshape__271 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__219, [x.reshape([]) for x in combine_135]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x64xf16) <- (-1x49x8x64xf16)
        transpose_169 = paddle._C_ops.transpose(reshape__270, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x1024xf16) <- (-1x49x512xf16, 512x1024xf16)
        matmul_183 = paddle._C_ops.matmul(layer_norm_243, parameter_492, False, False)

        # pd_op.add_: (-1x49x1024xf16) <- (-1x49x1024xf16, 1024xf16)
        add__220 = paddle._C_ops.add_(matmul_183, parameter_493)

        # pd_op.full: (1xi32) <- ()
        full_409 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_410 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_411 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_412 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_136 = [slice_83, full_409, full_410, full_411, full_412]

        # pd_op.reshape_: (-1x49x2x8x64xf16, 0x-1x49x1024xf16) <- (-1x49x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__272, reshape__273 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__220, [x.reshape([]) for x in combine_136]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x64xf16) <- (-1x49x2x8x64xf16)
        transpose_170 = paddle._C_ops.transpose(reshape__272, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_201 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [1]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(transpose_170, [0], full_int_array_201, full_int_array_202, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_204 = [2]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(transpose_170, [0], full_int_array_203, full_int_array_204, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_171 = paddle._C_ops.transpose(slice_84, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_184 = paddle._C_ops.matmul(transpose_169, transpose_171, False, False)

        # pd_op.full: (1xf32) <- ()
        full_413 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__26 = paddle._C_ops.scale_(matmul_184, full_413, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__26 = paddle._C_ops.softmax_(scale__26, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_185 = paddle._C_ops.matmul(softmax__26, slice_85, False, False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_172 = paddle._C_ops.transpose(matmul_185, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_414 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_415 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_137 = [slice_83, full_414, full_415]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__274, reshape__275 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_172, [x.reshape([]) for x in combine_137]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_186 = paddle._C_ops.matmul(reshape__274, parameter_494, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__221 = paddle._C_ops.add_(matmul_186, parameter_495)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__222 = paddle._C_ops.add_(transpose_168, add__221)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_246, layer_norm_247, layer_norm_248 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__222, parameter_496, parameter_497, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_187 = paddle._C_ops.matmul(layer_norm_246, parameter_498, False, False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__223 = paddle._C_ops.add_(matmul_187, parameter_499)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_26 = paddle._C_ops.gelu(add__223, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_188 = paddle._C_ops.matmul(gelu_26, parameter_500, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__224 = paddle._C_ops.add_(matmul_188, parameter_501)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__225 = paddle._C_ops.add_(add__222, add__224)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_249, layer_norm_250, layer_norm_251 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__225, parameter_502, parameter_503, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x49x512xf16)
        shape_32 = paddle._C_ops.shape(paddle.cast(layer_norm_249, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_32, [0], full_int_array_205, full_int_array_206, [1], [0])

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_189 = paddle._C_ops.matmul(layer_norm_249, parameter_504, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__226 = paddle._C_ops.add_(matmul_189, parameter_505)

        # pd_op.full: (1xi32) <- ()
        full_416 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_417 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_418 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_138 = [slice_86, full_416, full_417, full_418]

        # pd_op.reshape_: (-1x49x8x64xf16, 0x-1x49x512xf16) <- (-1x49x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__276, reshape__277 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__226, [x.reshape([]) for x in combine_138]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x64xf16) <- (-1x49x8x64xf16)
        transpose_173 = paddle._C_ops.transpose(reshape__276, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x1024xf16) <- (-1x49x512xf16, 512x1024xf16)
        matmul_190 = paddle._C_ops.matmul(layer_norm_249, parameter_506, False, False)

        # pd_op.add_: (-1x49x1024xf16) <- (-1x49x1024xf16, 1024xf16)
        add__227 = paddle._C_ops.add_(matmul_190, parameter_507)

        # pd_op.full: (1xi32) <- ()
        full_419 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_420 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_421 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_422 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_139 = [slice_86, full_419, full_420, full_421, full_422]

        # pd_op.reshape_: (-1x49x2x8x64xf16, 0x-1x49x1024xf16) <- (-1x49x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__278, reshape__279 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__227, [x.reshape([]) for x in combine_139]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x8x49x64xf16) <- (-1x49x2x8x64xf16)
        transpose_174 = paddle._C_ops.transpose(reshape__278, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [1]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_174, [0], full_int_array_207, full_int_array_208, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [2]

        # pd_op.slice: (-1x8x49x64xf16) <- (2x-1x8x49x64xf16, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(transpose_174, [0], full_int_array_209, full_int_array_210, [1], [0])

        # pd_op.transpose: (-1x8x64x49xf16) <- (-1x8x49x64xf16)
        transpose_175 = paddle._C_ops.transpose(slice_87, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x49xf16) <- (-1x8x49x64xf16, -1x8x64x49xf16)
        matmul_191 = paddle._C_ops.matmul(transpose_173, transpose_175, False, False)

        # pd_op.full: (1xf32) <- ()
        full_423 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x49xf16) <- (-1x8x49x49xf16, 1xf32)
        scale__27 = paddle._C_ops.scale_(matmul_191, full_423, float('0'), True)

        # pd_op.softmax_: (-1x8x49x49xf16) <- (-1x8x49x49xf16)
        softmax__27 = paddle._C_ops.softmax_(scale__27, -1)

        # pd_op.matmul: (-1x8x49x64xf16) <- (-1x8x49x49xf16, -1x8x49x64xf16)
        matmul_192 = paddle._C_ops.matmul(softmax__27, slice_88, False, False)

        # pd_op.transpose: (-1x49x8x64xf16) <- (-1x8x49x64xf16)
        transpose_176 = paddle._C_ops.transpose(matmul_192, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_424 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_425 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_140 = [slice_86, full_424, full_425]

        # pd_op.reshape_: (-1x49x512xf16, 0x-1x49x8x64xf16) <- (-1x49x8x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__280, reshape__281 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_176, [x.reshape([]) for x in combine_140]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x512xf16, 512x512xf16)
        matmul_193 = paddle._C_ops.matmul(reshape__280, parameter_508, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__228 = paddle._C_ops.add_(matmul_193, parameter_509)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__229 = paddle._C_ops.add_(add__225, add__228)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_252, layer_norm_253, layer_norm_254 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__229, parameter_510, parameter_511, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x49x2048xf16) <- (-1x49x512xf16, 512x2048xf16)
        matmul_194 = paddle._C_ops.matmul(layer_norm_252, parameter_512, False, False)

        # pd_op.add_: (-1x49x2048xf16) <- (-1x49x2048xf16, 2048xf16)
        add__230 = paddle._C_ops.add_(matmul_194, parameter_513)

        # pd_op.gelu: (-1x49x2048xf16) <- (-1x49x2048xf16)
        gelu_27 = paddle._C_ops.gelu(add__230, False)

        # pd_op.matmul: (-1x49x512xf16) <- (-1x49x2048xf16, 2048x512xf16)
        matmul_195 = paddle._C_ops.matmul(gelu_27, parameter_514, False, False)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, 512xf16)
        add__231 = paddle._C_ops.add_(matmul_195, parameter_515)

        # pd_op.add_: (-1x49x512xf16) <- (-1x49x512xf16, -1x49x512xf16)
        add__232 = paddle._C_ops.add_(add__229, add__231)

        # pd_op.layer_norm: (-1x49x512xf16, -49xf32, -49xf32) <- (-1x49x512xf16, 512xf32, 512xf32)
        layer_norm_255, layer_norm_256, layer_norm_257 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__232, parameter_516, parameter_517, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.mean: (-1x512xf16) <- (-1x49x512xf16)
        mean_0 = paddle._C_ops.mean(layer_norm_255, [1], False)

        # pd_op.matmul: (-1x1000xf16) <- (-1x512xf16, 512x1000xf16)
        matmul_196 = paddle._C_ops.matmul(mean_0, parameter_518, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__233 = paddle._C_ops.add_(matmul_196, parameter_519)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__28 = paddle._C_ops.softmax_(add__233, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__28, paddle.float32)
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

    def forward(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_46, parameter_47, parameter_49, parameter_48, parameter_50, parameter_51, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_95, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_125, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_185, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_215, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_245, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_305, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_349, parameter_348, parameter_350, parameter_351, parameter_352, parameter_353, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_361, parameter_360, parameter_362, parameter_363, parameter_364, parameter_365, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_373, parameter_372, parameter_374, parameter_375, parameter_376, parameter_377, parameter_379, parameter_378, parameter_380, parameter_381, parameter_382, parameter_383, parameter_385, parameter_384, parameter_386, parameter_387, parameter_388, parameter_389, parameter_391, parameter_390, parameter_392, parameter_393, parameter_394, parameter_395, parameter_397, parameter_396, parameter_398, parameter_399, parameter_400, parameter_401, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_409, parameter_408, parameter_410, parameter_411, parameter_412, parameter_413, parameter_415, parameter_414, parameter_416, parameter_417, parameter_418, parameter_419, parameter_421, parameter_420, parameter_422, parameter_423, parameter_424, parameter_425, parameter_427, parameter_426, parameter_428, parameter_429, parameter_430, parameter_431, parameter_433, parameter_432, parameter_434, parameter_435, parameter_436, parameter_437, parameter_439, parameter_438, parameter_440, parameter_441, parameter_442, parameter_443, parameter_445, parameter_444, parameter_446, parameter_447, parameter_448, parameter_449, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_457, parameter_456, parameter_458, parameter_459, parameter_460, parameter_461, parameter_463, parameter_462, parameter_464, parameter_465, parameter_466, parameter_467, parameter_468, parameter_469, parameter_471, parameter_470, parameter_473, parameter_472, parameter_474, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_481, parameter_480, parameter_482, parameter_483, parameter_484, parameter_485, parameter_486, parameter_487, parameter_489, parameter_488, parameter_490, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_497, parameter_496, parameter_498, parameter_499, parameter_500, parameter_501, parameter_503, parameter_502, parameter_504, parameter_505, parameter_506, parameter_507, parameter_508, parameter_509, parameter_511, parameter_510, parameter_512, parameter_513, parameter_514, parameter_515, parameter_517, parameter_516, parameter_518, parameter_519, feed_0):
        return self.builtin_module_2424_0_0(parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_14, parameter_15, parameter_17, parameter_16, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_25, parameter_24, parameter_26, parameter_27, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_34, parameter_35, parameter_37, parameter_36, parameter_38, parameter_39, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_46, parameter_47, parameter_49, parameter_48, parameter_50, parameter_51, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_63, parameter_62, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_74, parameter_75, parameter_77, parameter_76, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_85, parameter_84, parameter_86, parameter_87, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, parameter_94, parameter_95, parameter_97, parameter_96, parameter_98, parameter_99, parameter_100, parameter_101, parameter_103, parameter_102, parameter_104, parameter_105, parameter_106, parameter_107, parameter_109, parameter_108, parameter_110, parameter_111, parameter_112, parameter_113, parameter_115, parameter_114, parameter_116, parameter_117, parameter_118, parameter_119, parameter_121, parameter_120, parameter_122, parameter_123, parameter_124, parameter_125, parameter_127, parameter_126, parameter_128, parameter_129, parameter_130, parameter_131, parameter_133, parameter_132, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_141, parameter_140, parameter_143, parameter_142, parameter_144, parameter_145, parameter_146, parameter_147, parameter_149, parameter_148, parameter_150, parameter_151, parameter_152, parameter_153, parameter_155, parameter_154, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_171, parameter_172, parameter_173, parameter_175, parameter_174, parameter_176, parameter_177, parameter_178, parameter_179, parameter_181, parameter_180, parameter_182, parameter_183, parameter_184, parameter_185, parameter_187, parameter_186, parameter_188, parameter_189, parameter_190, parameter_191, parameter_193, parameter_192, parameter_194, parameter_195, parameter_196, parameter_197, parameter_199, parameter_198, parameter_200, parameter_201, parameter_202, parameter_203, parameter_205, parameter_204, parameter_206, parameter_207, parameter_208, parameter_209, parameter_211, parameter_210, parameter_212, parameter_213, parameter_214, parameter_215, parameter_217, parameter_216, parameter_218, parameter_219, parameter_220, parameter_221, parameter_223, parameter_222, parameter_224, parameter_225, parameter_226, parameter_227, parameter_229, parameter_228, parameter_230, parameter_231, parameter_232, parameter_233, parameter_235, parameter_234, parameter_236, parameter_237, parameter_238, parameter_239, parameter_241, parameter_240, parameter_242, parameter_243, parameter_244, parameter_245, parameter_247, parameter_246, parameter_248, parameter_249, parameter_250, parameter_251, parameter_253, parameter_252, parameter_254, parameter_255, parameter_256, parameter_257, parameter_259, parameter_258, parameter_260, parameter_261, parameter_262, parameter_263, parameter_265, parameter_264, parameter_266, parameter_267, parameter_268, parameter_269, parameter_271, parameter_270, parameter_272, parameter_273, parameter_274, parameter_275, parameter_277, parameter_276, parameter_278, parameter_279, parameter_280, parameter_281, parameter_283, parameter_282, parameter_284, parameter_285, parameter_286, parameter_287, parameter_289, parameter_288, parameter_290, parameter_291, parameter_292, parameter_293, parameter_295, parameter_294, parameter_296, parameter_297, parameter_298, parameter_299, parameter_301, parameter_300, parameter_302, parameter_303, parameter_304, parameter_305, parameter_307, parameter_306, parameter_308, parameter_309, parameter_310, parameter_311, parameter_313, parameter_312, parameter_314, parameter_315, parameter_316, parameter_317, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_345, parameter_346, parameter_347, parameter_349, parameter_348, parameter_350, parameter_351, parameter_352, parameter_353, parameter_355, parameter_354, parameter_356, parameter_357, parameter_358, parameter_359, parameter_361, parameter_360, parameter_362, parameter_363, parameter_364, parameter_365, parameter_367, parameter_366, parameter_368, parameter_369, parameter_370, parameter_371, parameter_373, parameter_372, parameter_374, parameter_375, parameter_376, parameter_377, parameter_379, parameter_378, parameter_380, parameter_381, parameter_382, parameter_383, parameter_385, parameter_384, parameter_386, parameter_387, parameter_388, parameter_389, parameter_391, parameter_390, parameter_392, parameter_393, parameter_394, parameter_395, parameter_397, parameter_396, parameter_398, parameter_399, parameter_400, parameter_401, parameter_403, parameter_402, parameter_404, parameter_405, parameter_406, parameter_407, parameter_409, parameter_408, parameter_410, parameter_411, parameter_412, parameter_413, parameter_415, parameter_414, parameter_416, parameter_417, parameter_418, parameter_419, parameter_421, parameter_420, parameter_422, parameter_423, parameter_424, parameter_425, parameter_427, parameter_426, parameter_428, parameter_429, parameter_430, parameter_431, parameter_433, parameter_432, parameter_434, parameter_435, parameter_436, parameter_437, parameter_439, parameter_438, parameter_440, parameter_441, parameter_442, parameter_443, parameter_445, parameter_444, parameter_446, parameter_447, parameter_448, parameter_449, parameter_451, parameter_450, parameter_452, parameter_453, parameter_454, parameter_455, parameter_457, parameter_456, parameter_458, parameter_459, parameter_460, parameter_461, parameter_463, parameter_462, parameter_464, parameter_465, parameter_466, parameter_467, parameter_468, parameter_469, parameter_471, parameter_470, parameter_473, parameter_472, parameter_474, parameter_475, parameter_476, parameter_477, parameter_478, parameter_479, parameter_481, parameter_480, parameter_482, parameter_483, parameter_484, parameter_485, parameter_486, parameter_487, parameter_489, parameter_488, parameter_490, parameter_491, parameter_492, parameter_493, parameter_494, parameter_495, parameter_497, parameter_496, parameter_498, parameter_499, parameter_500, parameter_501, parameter_503, parameter_502, parameter_504, parameter_505, parameter_506, parameter_507, parameter_508, parameter_509, parameter_511, parameter_510, parameter_512, parameter_513, parameter_514, parameter_515, parameter_517, parameter_516, parameter_518, parameter_519, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2424_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64, 64, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64, 128], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64, 512], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([512, 64], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64, 64, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64, 128], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64, 512], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([512, 64], dtype='float16', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64, 64, 8, 8], dtype='float16', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 128], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 64], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64, 512], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([512, 64], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128, 64, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([128, 128, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128, 256], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128, 1024], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1024, 128], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128, 128, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128, 256], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128, 1024], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([1024, 128], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128, 128, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([128, 256], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128, 1024], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1024, 128], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128, 128, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128, 256], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128, 1024], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([1024, 128], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([320, 128, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_185
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_247
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_255
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_262
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_331
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_346
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_347
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_351
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_361
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_363
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_365
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_367
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_370
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_371
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_375
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_376
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_377
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_381
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_382
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_383
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_385
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_387
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_388
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_389
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_391
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_393
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_395
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_397
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_400
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_401
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_403
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_405
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_406
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_407
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_409
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_411
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_412
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_413
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_415
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_417
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_418
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_419
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_421
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_423
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_424
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_425
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_427
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_429
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_430
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_431
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_433
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_435
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_436
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_437
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_439
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_441
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_442
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_443
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_445
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_447
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_448
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_449
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_451
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_453
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_454
            paddle.uniform([320, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_455
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_457
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([320, 640], dtype='float16', min=0, max=0.5),
            # parameter_459
            paddle.uniform([640], dtype='float16', min=0, max=0.5),
            # parameter_460
            paddle.uniform([320, 320], dtype='float16', min=0, max=0.5),
            # parameter_461
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_463
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([320, 1280], dtype='float16', min=0, max=0.5),
            # parameter_465
            paddle.uniform([1280], dtype='float16', min=0, max=0.5),
            # parameter_466
            paddle.uniform([1280, 320], dtype='float16', min=0, max=0.5),
            # parameter_467
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            # parameter_468
            paddle.uniform([512, 320, 2, 2], dtype='float16', min=0, max=0.5),
            # parameter_469
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_471
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_475
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_476
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_477
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_478
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_479
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_481
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_483
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_484
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_485
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_486
            paddle.uniform([512, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_487
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_489
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_491
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_492
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_493
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_494
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_495
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_497
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_499
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_500
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_501
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_503
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_505
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_506
            paddle.uniform([512, 1024], dtype='float16', min=0, max=0.5),
            # parameter_507
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_508
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_509
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_511
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_513
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_514
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_515
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_517
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([512, 1000], dtype='float16', min=0, max=0.5),
            # parameter_519
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
            paddle.static.InputSpec(shape=[64, 3, 4, 4], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[64, 64, 8, 8], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64, 128], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64, 512], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[512, 64], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[64, 64, 8, 8], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64, 128], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64, 512], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[512, 64], dtype='float16'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[64, 64, 8, 8], dtype='float16'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 128], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 64], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64, 512], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[512, 64], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[128, 64, 2, 2], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[128, 128, 4, 4], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128, 256], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128, 1024], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1024, 128], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[128, 128, 4, 4], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128, 256], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128, 1024], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[1024, 128], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[128, 128, 4, 4], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[128, 256], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128, 1024], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[1024, 128], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[128, 128, 4, 4], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128, 256], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128, 1024], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[1024, 128], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[320, 128, 2, 2], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_185
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_247
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_255
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_262
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_331
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_346
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_347
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_351
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_361
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_363
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_365
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_367
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_370
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_371
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_375
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_376
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_377
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_381
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_382
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_383
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_385
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_387
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_388
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_389
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_391
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_393
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_395
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_397
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_400
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_401
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_403
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_405
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_406
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_407
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_409
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_411
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_412
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_413
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_415
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_417
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_418
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_419
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_421
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_423
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_424
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_425
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_427
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_429
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_430
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_431
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_433
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_435
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_436
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_437
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_439
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_441
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_442
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_443
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_445
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_447
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_448
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_449
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_451
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_453
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_454
            paddle.static.InputSpec(shape=[320, 320, 2, 2], dtype='float16'),
            # parameter_455
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_457
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[320, 640], dtype='float16'),
            # parameter_459
            paddle.static.InputSpec(shape=[640], dtype='float16'),
            # parameter_460
            paddle.static.InputSpec(shape=[320, 320], dtype='float16'),
            # parameter_461
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_463
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[320, 1280], dtype='float16'),
            # parameter_465
            paddle.static.InputSpec(shape=[1280], dtype='float16'),
            # parameter_466
            paddle.static.InputSpec(shape=[1280, 320], dtype='float16'),
            # parameter_467
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            # parameter_468
            paddle.static.InputSpec(shape=[512, 320, 2, 2], dtype='float16'),
            # parameter_469
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_471
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_475
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_476
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_477
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_478
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_479
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_481
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_483
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_484
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_485
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_486
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float16'),
            # parameter_487
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_489
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_491
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_492
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_493
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_494
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_495
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_497
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_499
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_500
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_501
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_503
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_505
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_506
            paddle.static.InputSpec(shape=[512, 1024], dtype='float16'),
            # parameter_507
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_508
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_509
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_511
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_513
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_514
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_515
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_517
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[512, 1000], dtype='float16'),
            # parameter_519
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