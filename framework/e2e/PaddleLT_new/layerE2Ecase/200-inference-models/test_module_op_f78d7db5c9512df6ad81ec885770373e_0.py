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
    return [922][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1318_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_381, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_388, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x3x224x224xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x32x112x112xf16, 32x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x32x112x112xf16, 64x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf16) <- (-1x64x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 64x64x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf16) <- (-1x64x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x64x56x56xf16, 128x32x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x56x56xf16, -1x64x56x56xf16]) <- (-1x128x56x56xf16, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(relu__4, 2, full_0)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_0 = split_with_num_0[0]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_1 = paddle._C_ops.cast(slice_0, paddle.float32)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_1 = split_with_num_0[1]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_2 = paddle._C_ops.cast(slice_1, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_0 = [cast_1, cast_2]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_0 = paddle._C_ops.add_n(combine_0)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_3 = paddle._C_ops.cast(add_n_0, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf16) <- (-1x64x56x56xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(cast_3, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf16) <- (-1x64x1x1xf16, 32x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf16) <- (-1x32x1x1xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x32x1x1xf16, 128x32x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_31, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_6, reshape_0)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(add__0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_2, full_1, full_2, full_3]

        # pd_op.reshape_: (-1x1x2x64xf16, 0x-1x128x1x1xf16) <- (-1x128x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf16) <- (-1x1x2x64xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf16) <- (-1x2x1x64xf16)
        softmax__0 = paddle._C_ops.softmax_(transpose_0, 1)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_2, full_4, full_5, full_6]

        # pd_op.reshape_: (-1x128x1x1xf16, 0x-1x2x1x64xf16) <- (-1x2x1x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__0, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf16, -1x64x1x1xf16]) <- (-1x128x1x1xf16, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(reshape__2, 2, full_7)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_3 = split_with_num_1[0]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__0 = paddle._C_ops.multiply_(slice_0, slice_3)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_4 = split_with_num_1[1]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__1 = paddle._C_ops.multiply_(slice_1, slice_4)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_4 = paddle._C_ops.cast(multiply__0, paddle.float32)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_5 = paddle._C_ops.cast(multiply__1, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_3 = [cast_4, cast_5]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_1 = paddle._C_ops.add_n(combine_3)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_6 = paddle._C_ops.cast(add_n_1, paddle.float16)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(cast_6, parameter_32, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x64x56x56xf16) <- (-1x64x56x56xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_0, full_int_array_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_2, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__1 = paddle._C_ops.add_(batch_norm__42, batch_norm__36)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__6 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__6, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf16) <- (-1x64x56x56xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x64x56x56xf16, 128x32x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__7, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x56x56xf16, -1x64x56x56xf16]) <- (-1x128x56x56xf16, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(relu__8, 2, full_8)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_5 = split_with_num_2[0]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_7 = paddle._C_ops.cast(slice_5, paddle.float32)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_6 = split_with_num_2[1]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_8 = paddle._C_ops.cast(slice_6, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_4 = [cast_7, cast_8]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_2 = paddle._C_ops.add_n(combine_4)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_9 = paddle._C_ops.cast(add_n_2, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf16) <- (-1x64x56x56xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(cast_9, full_int_array_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf16) <- (-1x64x1x1xf16, 32x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_3, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf16) <- (-1x32x1x1xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x32x1x1xf16, 128x32x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__9, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_58, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_12, reshape_2)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(add__2, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_7, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x1x2x64xf16, 0x-1x128x1x1xf16) <- (-1x128x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf16) <- (-1x1x2x64xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf16) <- (-1x2x1x64xf16)
        softmax__1 = paddle._C_ops.softmax_(transpose_1, 1)

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_7, full_12, full_13, full_14]

        # pd_op.reshape_: (-1x128x1x1xf16, 0x-1x2x1x64xf16) <- (-1x2x1x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__1, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf16, -1x64x1x1xf16]) <- (-1x128x1x1xf16, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(reshape__6, 2, full_15)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_8 = split_with_num_3[0]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__2 = paddle._C_ops.multiply_(slice_5, slice_8)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_9 = split_with_num_3[1]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__3 = paddle._C_ops.multiply_(slice_6, slice_9)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_10 = paddle._C_ops.cast(multiply__2, paddle.float32)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_11 = paddle._C_ops.cast(multiply__3, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_7 = [cast_10, cast_11]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_3 = paddle._C_ops.add_n(combine_7)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_12 = paddle._C_ops.cast(add_n_3, paddle.float16)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(cast_12, parameter_59, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__3 = paddle._C_ops.add_(relu__6, batch_norm__66)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__10 = paddle._C_ops.relu_(add__3)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x256x56x56xf16, 64x256x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__10, parameter_64, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf16) <- (-1x64x56x56xf16)
        relu__11 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x64x56x56xf16, 128x32x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__11, parameter_69, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x56x56xf16, -1x64x56x56xf16]) <- (-1x128x56x56xf16, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(relu__12, 2, full_16)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_10 = split_with_num_4[0]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_13 = paddle._C_ops.cast(slice_10, paddle.float32)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16])
        slice_11 = split_with_num_4[1]

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_14 = paddle._C_ops.cast(slice_11, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_8 = [cast_13, cast_14]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_4 = paddle._C_ops.add_n(combine_8)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_15 = paddle._C_ops.cast(add_n_4, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf16) <- (-1x64x56x56xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(cast_15, full_int_array_10, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf16) <- (-1x64x1x1xf16, 32x64x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_4, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf16) <- (-1x32x1x1xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x32x1x1xf16, 128x32x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__13, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_80, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_17, reshape_4)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__4, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_2, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_12, full_17, full_18, full_19]

        # pd_op.reshape_: (-1x1x2x64xf16, 0x-1x128x1x1xf16) <- (-1x128x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__4, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf16) <- (-1x1x2x64xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf16) <- (-1x2x1x64xf16)
        softmax__2 = paddle._C_ops.softmax_(transpose_2, 1)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_12, full_20, full_21, full_22]

        # pd_op.reshape_: (-1x128x1x1xf16, 0x-1x2x1x64xf16) <- (-1x2x1x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__2, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf16, -1x64x1x1xf16]) <- (-1x128x1x1xf16, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(reshape__10, 2, full_23)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_13 = split_with_num_5[0]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__4 = paddle._C_ops.multiply_(slice_10, slice_13)

        # builtin.slice: (-1x64x1x1xf16) <- ([-1x64x1x1xf16, -1x64x1x1xf16])
        slice_14 = split_with_num_5[1]

        # pd_op.multiply_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x1x1xf16)
        multiply__5 = paddle._C_ops.multiply_(slice_11, slice_14)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_16 = paddle._C_ops.cast(multiply__4, paddle.float32)

        # pd_op.cast: (-1x64x56x56xf32) <- (-1x64x56x56xf16)
        cast_17 = paddle._C_ops.cast(multiply__5, paddle.float32)

        # builtin.combine: ([-1x64x56x56xf32, -1x64x56x56xf32]) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        combine_11 = [cast_16, cast_17]

        # pd_op.add_n: (-1x64x56x56xf32) <- ([-1x64x56x56xf32, -1x64x56x56xf32])
        add_n_5 = paddle._C_ops.add_n(combine_11)

        # pd_op.cast: (-1x64x56x56xf16) <- (-1x64x56x56xf32)
        cast_18 = paddle._C_ops.cast(add_n_5, paddle.float16)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(cast_18, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__5 = paddle._C_ops.add_(relu__10, batch_norm__90)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__14 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x256x56x56xf16, 128x256x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__14, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x128x56x56xf16, 256x64x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__15, parameter_91, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x56x56xf16, -1x128x56x56xf16]) <- (-1x256x56x56xf16, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(relu__16, 2, full_24)

        # builtin.slice: (-1x128x56x56xf16) <- ([-1x128x56x56xf16, -1x128x56x56xf16])
        slice_15 = split_with_num_6[0]

        # pd_op.cast: (-1x128x56x56xf32) <- (-1x128x56x56xf16)
        cast_19 = paddle._C_ops.cast(slice_15, paddle.float32)

        # builtin.slice: (-1x128x56x56xf16) <- ([-1x128x56x56xf16, -1x128x56x56xf16])
        slice_16 = split_with_num_6[1]

        # pd_op.cast: (-1x128x56x56xf32) <- (-1x128x56x56xf16)
        cast_20 = paddle._C_ops.cast(slice_16, paddle.float32)

        # builtin.combine: ([-1x128x56x56xf32, -1x128x56x56xf32]) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        combine_12 = [cast_19, cast_20]

        # pd_op.add_n: (-1x128x56x56xf32) <- ([-1x128x56x56xf32, -1x128x56x56xf32])
        add_n_6 = paddle._C_ops.add_n(combine_12)

        # pd_op.cast: (-1x128x56x56xf16) <- (-1x128x56x56xf32)
        cast_21 = paddle._C_ops.cast(add_n_6, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf16) <- (-1x128x56x56xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(cast_21, full_int_array_14, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf16) <- (-1x128x1x1xf16, 64x128x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(pool2d_5, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf16) <- (-1x64x1x1xf16)
        relu__17 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x64x1x1xf16, 256x64x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__17, parameter_101, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_102, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf16) <- (-1x256x1x1xf16, 1x256x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_22, reshape_6)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(add__6, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_3, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_17, full_25, full_26, full_27]

        # pd_op.reshape_: (-1x1x2x128xf16, 0x-1x256x1x1xf16) <- (-1x256x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__6, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf16) <- (-1x1x2x128xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf16) <- (-1x2x1x128xf16)
        softmax__3 = paddle._C_ops.softmax_(transpose_3, 1)

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_17, full_28, full_29, full_30]

        # pd_op.reshape_: (-1x256x1x1xf16, 0x-1x2x1x128xf16) <- (-1x2x1x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__3, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf16, -1x128x1x1xf16]) <- (-1x256x1x1xf16, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(reshape__14, 2, full_31)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_18 = split_with_num_7[0]

        # pd_op.multiply_: (-1x128x56x56xf16) <- (-1x128x56x56xf16, -1x128x1x1xf16)
        multiply__6 = paddle._C_ops.multiply_(slice_15, slice_18)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_19 = split_with_num_7[1]

        # pd_op.multiply_: (-1x128x56x56xf16) <- (-1x128x56x56xf16, -1x128x1x1xf16)
        multiply__7 = paddle._C_ops.multiply_(slice_16, slice_19)

        # pd_op.cast: (-1x128x56x56xf32) <- (-1x128x56x56xf16)
        cast_22 = paddle._C_ops.cast(multiply__6, paddle.float32)

        # pd_op.cast: (-1x128x56x56xf32) <- (-1x128x56x56xf16)
        cast_23 = paddle._C_ops.cast(multiply__7, paddle.float32)

        # builtin.combine: ([-1x128x56x56xf32, -1x128x56x56xf32]) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        combine_15 = [cast_22, cast_23]

        # pd_op.add_n: (-1x128x56x56xf32) <- ([-1x128x56x56xf32, -1x128x56x56xf32])
        add_n_7 = paddle._C_ops.add_n(combine_15)

        # pd_op.cast: (-1x128x56x56xf16) <- (-1x128x56x56xf32)
        cast_24 = paddle._C_ops.cast(add_n_7, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [3, 3]

        # pd_op.pool2d: (-1x128x28x28xf16) <- (-1x128x56x56xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(cast_24, full_int_array_18, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(pool2d_6, parameter_103, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_104, parameter_105, parameter_106, parameter_107, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [2, 2]

        # pd_op.pool2d: (-1x256x28x28xf16) <- (-1x256x56x56xf16, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(relu__14, full_int_array_19, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x256x28x28xf16, 512x256x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(pool2d_7, parameter_108, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_109, parameter_110, parameter_111, parameter_112, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__7 = paddle._C_ops.add_(batch_norm__120, batch_norm__114)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__18 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__18, parameter_113, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_114, parameter_115, parameter_116, parameter_117, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf16) <- (-1x128x28x28xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x128x28x28xf16, 256x64x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__19, parameter_118, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_119, parameter_120, parameter_121, parameter_122, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x28x28xf16, -1x128x28x28xf16]) <- (-1x256x28x28xf16, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(relu__20, 2, full_32)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_20 = split_with_num_8[0]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_25 = paddle._C_ops.cast(slice_20, paddle.float32)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_21 = split_with_num_8[1]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_26 = paddle._C_ops.cast(slice_21, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_16 = [cast_25, cast_26]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_8 = paddle._C_ops.add_n(combine_16)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_27 = paddle._C_ops.cast(add_n_8, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf16) <- (-1x128x28x28xf16, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(cast_27, full_int_array_20, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf16) <- (-1x128x1x1xf16, 64x128x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(pool2d_8, parameter_123, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_124, parameter_125, parameter_126, parameter_127, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf16) <- (-1x64x1x1xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x64x1x1xf16, 256x64x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__21, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_129, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf16) <- (-1x256x1x1xf16, 1x256x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_28, reshape_8)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(add__8, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_4, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_22, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x1x2x128xf16, 0x-1x256x1x1xf16) <- (-1x256x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf16) <- (-1x1x2x128xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf16) <- (-1x2x1x128xf16)
        softmax__4 = paddle._C_ops.softmax_(transpose_4, 1)

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_22, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x256x1x1xf16, 0x-1x2x1x128xf16) <- (-1x2x1x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__4, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf16, -1x128x1x1xf16]) <- (-1x256x1x1xf16, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(reshape__18, 2, full_39)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_23 = split_with_num_9[0]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__8 = paddle._C_ops.multiply_(slice_20, slice_23)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_24 = split_with_num_9[1]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__9 = paddle._C_ops.multiply_(slice_21, slice_24)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_28 = paddle._C_ops.cast(multiply__8, paddle.float32)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_29 = paddle._C_ops.cast(multiply__9, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_19 = [cast_28, cast_29]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_9 = paddle._C_ops.add_n(combine_19)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_30 = paddle._C_ops.cast(add_n_9, paddle.float16)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(cast_30, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__9 = paddle._C_ops.add_(relu__18, batch_norm__144)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__22 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__22, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf16) <- (-1x128x28x28xf16)
        relu__23 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x128x28x28xf16, 256x64x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__23, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x28x28xf16, -1x128x28x28xf16]) <- (-1x256x28x28xf16, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(relu__24, 2, full_40)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_25 = split_with_num_10[0]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_31 = paddle._C_ops.cast(slice_25, paddle.float32)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_26 = split_with_num_10[1]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_32 = paddle._C_ops.cast(slice_26, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_20 = [cast_31, cast_32]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_10 = paddle._C_ops.add_n(combine_20)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_33 = paddle._C_ops.cast(add_n_10, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf16) <- (-1x128x28x28xf16, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(cast_33, full_int_array_24, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf16) <- (-1x128x1x1xf16, 64x128x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(pool2d_9, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf16) <- (-1x64x1x1xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x64x1x1xf16, 256x64x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu__25, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf16) <- (-1x256x1x1xf16, 1x256x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_33, reshape_10)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add__10, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_5, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_27, full_41, full_42, full_43]

        # pd_op.reshape_: (-1x1x2x128xf16, 0x-1x256x1x1xf16) <- (-1x256x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf16) <- (-1x1x2x128xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf16) <- (-1x2x1x128xf16)
        softmax__5 = paddle._C_ops.softmax_(transpose_5, 1)

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_27, full_44, full_45, full_46]

        # pd_op.reshape_: (-1x256x1x1xf16, 0x-1x2x1x128xf16) <- (-1x2x1x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__5, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf16, -1x128x1x1xf16]) <- (-1x256x1x1xf16, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(reshape__22, 2, full_47)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_28 = split_with_num_11[0]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__10 = paddle._C_ops.multiply_(slice_25, slice_28)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_29 = split_with_num_11[1]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__11 = paddle._C_ops.multiply_(slice_26, slice_29)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_34 = paddle._C_ops.cast(multiply__10, paddle.float32)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_35 = paddle._C_ops.cast(multiply__11, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_23 = [cast_34, cast_35]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_11 = paddle._C_ops.add_n(combine_23)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_36 = paddle._C_ops.cast(add_n_11, paddle.float16)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(cast_36, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__11 = paddle._C_ops.add_(relu__22, batch_norm__168)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__26 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x128x28x28xf16) <- (-1x512x28x28xf16, 128x512x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__26, parameter_157, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf16) <- (-1x128x28x28xf16)
        relu__27 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x128x28x28xf16, 256x64x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu__27, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x28x28xf16, -1x128x28x28xf16]) <- (-1x256x28x28xf16, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(relu__28, 2, full_48)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_30 = split_with_num_12[0]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_37 = paddle._C_ops.cast(slice_30, paddle.float32)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16])
        slice_31 = split_with_num_12[1]

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_38 = paddle._C_ops.cast(slice_31, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_24 = [cast_37, cast_38]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_12 = paddle._C_ops.add_n(combine_24)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_39 = paddle._C_ops.cast(add_n_12, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf16) <- (-1x128x28x28xf16, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(cast_39, full_int_array_28, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf16) <- (-1x128x1x1xf16, 64x128x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(pool2d_10, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf16) <- (-1x64x1x1xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x64x1x1xf16, 256x64x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__29, parameter_172, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_173, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf16) <- (-1x256x1x1xf16, 1x256x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_38, reshape_12)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(add__12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_6, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_32, full_49, full_50, full_51]

        # pd_op.reshape_: (-1x1x2x128xf16, 0x-1x256x1x1xf16) <- (-1x256x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__12, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf16) <- (-1x1x2x128xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf16) <- (-1x2x1x128xf16)
        softmax__6 = paddle._C_ops.softmax_(transpose_6, 1)

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_32, full_52, full_53, full_54]

        # pd_op.reshape_: (-1x256x1x1xf16, 0x-1x2x1x128xf16) <- (-1x2x1x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__6, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf16, -1x128x1x1xf16]) <- (-1x256x1x1xf16, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(reshape__26, 2, full_55)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_33 = split_with_num_13[0]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__12 = paddle._C_ops.multiply_(slice_30, slice_33)

        # builtin.slice: (-1x128x1x1xf16) <- ([-1x128x1x1xf16, -1x128x1x1xf16])
        slice_34 = split_with_num_13[1]

        # pd_op.multiply_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x1x1xf16)
        multiply__13 = paddle._C_ops.multiply_(slice_31, slice_34)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_40 = paddle._C_ops.cast(multiply__12, paddle.float32)

        # pd_op.cast: (-1x128x28x28xf32) <- (-1x128x28x28xf16)
        cast_41 = paddle._C_ops.cast(multiply__13, paddle.float32)

        # builtin.combine: ([-1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        combine_27 = [cast_40, cast_41]

        # pd_op.add_n: (-1x128x28x28xf32) <- ([-1x128x28x28xf32, -1x128x28x28xf32])
        add_n_13 = paddle._C_ops.add_n(combine_27)

        # pd_op.cast: (-1x128x28x28xf16) <- (-1x128x28x28xf32)
        cast_42 = paddle._C_ops.cast(add_n_13, paddle.float16)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x128x28x28xf16, 512x128x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(cast_42, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__13 = paddle._C_ops.add_(relu__26, batch_norm__192)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__30 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x512x28x28xf16, 256x512x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__30, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__198)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x256x28x28xf16, 512x128x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__31, parameter_184, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__32 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x28x28xf16, -1x256x28x28xf16]) <- (-1x512x28x28xf16, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(relu__32, 2, full_56)

        # builtin.slice: (-1x256x28x28xf16) <- ([-1x256x28x28xf16, -1x256x28x28xf16])
        slice_35 = split_with_num_14[0]

        # pd_op.cast: (-1x256x28x28xf32) <- (-1x256x28x28xf16)
        cast_43 = paddle._C_ops.cast(slice_35, paddle.float32)

        # builtin.slice: (-1x256x28x28xf16) <- ([-1x256x28x28xf16, -1x256x28x28xf16])
        slice_36 = split_with_num_14[1]

        # pd_op.cast: (-1x256x28x28xf32) <- (-1x256x28x28xf16)
        cast_44 = paddle._C_ops.cast(slice_36, paddle.float32)

        # builtin.combine: ([-1x256x28x28xf32, -1x256x28x28xf32]) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        combine_28 = [cast_43, cast_44]

        # pd_op.add_n: (-1x256x28x28xf32) <- ([-1x256x28x28xf32, -1x256x28x28xf32])
        add_n_14 = paddle._C_ops.add_n(combine_28)

        # pd_op.cast: (-1x256x28x28xf16) <- (-1x256x28x28xf32)
        cast_45 = paddle._C_ops.cast(add_n_14, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x28x28xf16, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(cast_45, full_int_array_32, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(pool2d_11, parameter_189, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__33, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_43, reshape_14)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(add__14, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(shape_7, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_37, full_57, full_58, full_59]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__7 = paddle._C_ops.softmax_(transpose_7, 1)

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_37, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__7, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(reshape__30, 2, full_63)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_38 = split_with_num_15[0]

        # pd_op.multiply_: (-1x256x28x28xf16) <- (-1x256x28x28xf16, -1x256x1x1xf16)
        multiply__14 = paddle._C_ops.multiply_(slice_35, slice_38)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_39 = split_with_num_15[1]

        # pd_op.multiply_: (-1x256x28x28xf16) <- (-1x256x28x28xf16, -1x256x1x1xf16)
        multiply__15 = paddle._C_ops.multiply_(slice_36, slice_39)

        # pd_op.cast: (-1x256x28x28xf32) <- (-1x256x28x28xf16)
        cast_46 = paddle._C_ops.cast(multiply__14, paddle.float32)

        # pd_op.cast: (-1x256x28x28xf32) <- (-1x256x28x28xf16)
        cast_47 = paddle._C_ops.cast(multiply__15, paddle.float32)

        # builtin.combine: ([-1x256x28x28xf32, -1x256x28x28xf32]) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        combine_31 = [cast_46, cast_47]

        # pd_op.add_n: (-1x256x28x28xf32) <- ([-1x256x28x28xf32, -1x256x28x28xf32])
        add_n_15 = paddle._C_ops.add_n(combine_31)

        # pd_op.cast: (-1x256x28x28xf16) <- (-1x256x28x28xf32)
        cast_48 = paddle._C_ops.cast(add_n_15, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [3, 3]

        # pd_op.pool2d: (-1x256x14x14xf16) <- (-1x256x28x28xf16, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(cast_48, full_int_array_36, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(pool2d_12, parameter_196, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [2, 2]

        # pd_op.pool2d: (-1x512x14x14xf16) <- (-1x512x28x28xf16, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(relu__30, full_int_array_37, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_13, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__15 = paddle._C_ops.add_(batch_norm__222, batch_norm__216)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__34 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x1024x14x14xf16, 256x1024x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__34, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_207, parameter_208, parameter_209, parameter_210, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__35 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x128x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__35, parameter_211, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x512x14x14xf16, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(relu__36, 2, full_64)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_40 = split_with_num_16[0]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_49 = paddle._C_ops.cast(slice_40, paddle.float32)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_41 = split_with_num_16[1]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_50 = paddle._C_ops.cast(slice_41, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_32 = [cast_49, cast_50]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_16 = paddle._C_ops.add_n(combine_32)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_51 = paddle._C_ops.cast(add_n_16, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(cast_51, full_int_array_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(pool2d_14, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_217, parameter_218, parameter_219, parameter_220, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__37 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__37, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_222, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_49, reshape_16)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add__16, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_8, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_42, full_65, full_66, full_67]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__16, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__8 = paddle._C_ops.softmax_(transpose_8, 1)

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_42, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__8, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(reshape__34, 2, full_71)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_43 = split_with_num_17[0]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__16 = paddle._C_ops.multiply_(slice_40, slice_43)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_44 = split_with_num_17[1]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__17 = paddle._C_ops.multiply_(slice_41, slice_44)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_52 = paddle._C_ops.cast(multiply__16, paddle.float32)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_53 = paddle._C_ops.cast(multiply__17, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_35 = [cast_52, cast_53]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_17 = paddle._C_ops.add_n(combine_35)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_54 = paddle._C_ops.cast(add_n_17, paddle.float16)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(cast_54, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__17 = paddle._C_ops.add_(relu__34, batch_norm__246)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__38 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x1024x14x14xf16, 256x1024x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__38, parameter_228, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__39 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x128x3x3xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__39, parameter_233, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_234, parameter_235, parameter_236, parameter_237, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x512x14x14xf16, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(relu__40, 2, full_72)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_45 = split_with_num_18[0]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_55 = paddle._C_ops.cast(slice_45, paddle.float32)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_46 = split_with_num_18[1]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_56 = paddle._C_ops.cast(slice_46, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_36 = [cast_55, cast_56]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_18 = paddle._C_ops.add_n(combine_36)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_57 = paddle._C_ops.cast(add_n_18, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(cast_57, full_int_array_42, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(pool2d_15, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_239, parameter_240, parameter_241, parameter_242, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__41 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__41, parameter_243, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_244, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_54, reshape_18)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(add__18, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_9, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_47, full_73, full_74, full_75]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__9 = paddle._C_ops.softmax_(transpose_9, 1)

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_47, full_76, full_77, full_78]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__9, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(reshape__38, 2, full_79)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_48 = split_with_num_19[0]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__18 = paddle._C_ops.multiply_(slice_45, slice_48)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_49 = split_with_num_19[1]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__19 = paddle._C_ops.multiply_(slice_46, slice_49)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_58 = paddle._C_ops.cast(multiply__18, paddle.float32)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_59 = paddle._C_ops.cast(multiply__19, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_39 = [cast_58, cast_59]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_19 = paddle._C_ops.add_n(combine_39)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_60 = paddle._C_ops.cast(add_n_19, paddle.float16)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(cast_60, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__19 = paddle._C_ops.add_(relu__38, batch_norm__270)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__42 = paddle._C_ops.relu_(add__19)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x1024x14x14xf16, 256x1024x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__42, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x128x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu__43, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x512x14x14xf16, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(relu__44, 2, full_80)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_50 = split_with_num_20[0]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_61 = paddle._C_ops.cast(slice_50, paddle.float32)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_51 = split_with_num_20[1]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_62 = paddle._C_ops.cast(slice_51, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_40 = [cast_61, cast_62]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_20 = paddle._C_ops.add_n(combine_40)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_63 = paddle._C_ops.cast(add_n_20, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_46 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_16 = paddle._C_ops.pool2d(cast_63, full_int_array_46, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(pool2d_16, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu__45, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_266, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_59, reshape_20)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(add__20, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_10, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_52, full_81, full_82, full_83]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__40, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__10 = paddle._C_ops.softmax_(transpose_10, 1)

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_52, full_84, full_85, full_86]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__10, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(reshape__42, 2, full_87)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_53 = split_with_num_21[0]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__20 = paddle._C_ops.multiply_(slice_50, slice_53)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_54 = split_with_num_21[1]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__21 = paddle._C_ops.multiply_(slice_51, slice_54)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_64 = paddle._C_ops.cast(multiply__20, paddle.float32)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_65 = paddle._C_ops.cast(multiply__21, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_43 = [cast_64, cast_65]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_21 = paddle._C_ops.add_n(combine_43)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_66 = paddle._C_ops.cast(add_n_21, paddle.float16)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(cast_66, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__21 = paddle._C_ops.add_(relu__42, batch_norm__294)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__46 = paddle._C_ops.relu_(add__21)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x1024x14x14xf16, 256x1024x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(relu__46, parameter_272, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__47 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x128x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu__47, parameter_277, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_278, parameter_279, parameter_280, parameter_281, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x512x14x14xf16, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(relu__48, 2, full_88)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_55 = split_with_num_22[0]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_67 = paddle._C_ops.cast(slice_55, paddle.float32)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_56 = split_with_num_22[1]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_68 = paddle._C_ops.cast(slice_56, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_44 = [cast_67, cast_68]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_22 = paddle._C_ops.add_n(combine_44)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_69 = paddle._C_ops.cast(add_n_22, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_17 = paddle._C_ops.pool2d(cast_69, full_int_array_50, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(pool2d_17, parameter_282, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_283, parameter_284, parameter_285, parameter_286, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu__49, parameter_287, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_64, reshape_22)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(add__22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_11, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_57, full_89, full_90, full_91]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__11 = paddle._C_ops.softmax_(transpose_11, 1)

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_57, full_92, full_93, full_94]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__11, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(reshape__46, 2, full_95)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_58 = split_with_num_23[0]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__22 = paddle._C_ops.multiply_(slice_55, slice_58)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_59 = split_with_num_23[1]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__23 = paddle._C_ops.multiply_(slice_56, slice_59)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_70 = paddle._C_ops.cast(multiply__22, paddle.float32)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_71 = paddle._C_ops.cast(multiply__23, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_47 = [cast_70, cast_71]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_23 = paddle._C_ops.add_n(combine_47)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_72 = paddle._C_ops.cast(add_n_23, paddle.float16)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(cast_72, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_290, parameter_291, parameter_292, parameter_293, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__23 = paddle._C_ops.add_(relu__46, batch_norm__318)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__50 = paddle._C_ops.relu_(add__23)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x1024x14x14xf16, 256x1024x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu__50, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__51 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x128x3x3xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__51, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__52 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x512x14x14xf16, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(relu__52, 2, full_96)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_60 = split_with_num_24[0]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_73 = paddle._C_ops.cast(slice_60, paddle.float32)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16])
        slice_61 = split_with_num_24[1]

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_74 = paddle._C_ops.cast(slice_61, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_48 = [cast_73, cast_74]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_24 = paddle._C_ops.add_n(combine_48)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_75 = paddle._C_ops.cast(add_n_24, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_54 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_18 = paddle._C_ops.pool2d(cast_75, full_int_array_54, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x256x1x1xf16, 128x256x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(pool2d_18, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__53 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__53, parameter_309, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_310, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__24 = paddle._C_ops.add_(conv2d_69, reshape_24)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(add__24, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_12, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_62, full_97, full_98, full_99]

        # pd_op.reshape_: (-1x1x2x256xf16, 0x-1x512x1x1xf16) <- (-1x512x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__24, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf16) <- (-1x1x2x256xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf16) <- (-1x2x1x256xf16)
        softmax__12 = paddle._C_ops.softmax_(transpose_12, 1)

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_62, full_100, full_101, full_102]

        # pd_op.reshape_: (-1x512x1x1xf16, 0x-1x2x1x256xf16) <- (-1x2x1x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__12, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf16, -1x256x1x1xf16]) <- (-1x512x1x1xf16, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(reshape__50, 2, full_103)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_63 = split_with_num_25[0]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__24 = paddle._C_ops.multiply_(slice_60, slice_63)

        # builtin.slice: (-1x256x1x1xf16) <- ([-1x256x1x1xf16, -1x256x1x1xf16])
        slice_64 = split_with_num_25[1]

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__25 = paddle._C_ops.multiply_(slice_61, slice_64)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_76 = paddle._C_ops.cast(multiply__24, paddle.float32)

        # pd_op.cast: (-1x256x14x14xf32) <- (-1x256x14x14xf16)
        cast_77 = paddle._C_ops.cast(multiply__25, paddle.float32)

        # builtin.combine: ([-1x256x14x14xf32, -1x256x14x14xf32]) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        combine_51 = [cast_76, cast_77]

        # pd_op.add_n: (-1x256x14x14xf32) <- ([-1x256x14x14xf32, -1x256x14x14xf32])
        add_n_25 = paddle._C_ops.add_n(combine_51)

        # pd_op.cast: (-1x256x14x14xf16) <- (-1x256x14x14xf32)
        cast_78 = paddle._C_ops.cast(add_n_25, paddle.float16)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x256x14x14xf16, 1024x256x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(cast_78, parameter_311, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__25 = paddle._C_ops.add_(relu__50, batch_norm__342)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__54 = paddle._C_ops.relu_(add__25)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x1024x14x14xf16, 512x1024x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(relu__54, parameter_316, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_317, parameter_318, parameter_319, parameter_320, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__55 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x256x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(relu__55, parameter_321, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_322, parameter_323, parameter_324, parameter_325, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x14x14xf16, -1x512x14x14xf16]) <- (-1x1024x14x14xf16, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(relu__56, 2, full_104)

        # builtin.slice: (-1x512x14x14xf16) <- ([-1x512x14x14xf16, -1x512x14x14xf16])
        slice_65 = split_with_num_26[0]

        # pd_op.cast: (-1x512x14x14xf32) <- (-1x512x14x14xf16)
        cast_79 = paddle._C_ops.cast(slice_65, paddle.float32)

        # builtin.slice: (-1x512x14x14xf16) <- ([-1x512x14x14xf16, -1x512x14x14xf16])
        slice_66 = split_with_num_26[1]

        # pd_op.cast: (-1x512x14x14xf32) <- (-1x512x14x14xf16)
        cast_80 = paddle._C_ops.cast(slice_66, paddle.float32)

        # builtin.combine: ([-1x512x14x14xf32, -1x512x14x14xf32]) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        combine_52 = [cast_79, cast_80]

        # pd_op.add_n: (-1x512x14x14xf32) <- ([-1x512x14x14xf32, -1x512x14x14xf32])
        add_n_26 = paddle._C_ops.add_n(combine_52)

        # pd_op.cast: (-1x512x14x14xf16) <- (-1x512x14x14xf32)
        cast_81 = paddle._C_ops.cast(add_n_26, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_19 = paddle._C_ops.pool2d(cast_81, full_int_array_58, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x512x1x1xf16, 256x512x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(pool2d_19, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_327, parameter_328, parameter_329, parameter_330, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf16) <- (-1x256x1x1xf16)
        relu__57 = paddle._C_ops.relu_(batch_norm__360)

        # pd_op.conv2d: (-1x1024x1x1xf16) <- (-1x256x1x1xf16, 1024x256x1x1xf16)
        conv2d_74 = paddle._C_ops.conv2d(relu__57, parameter_331, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_332, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf16) <- (-1x1024x1x1xf16, 1x1024x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_74, reshape_26)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(add__26, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(shape_13, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_67, full_105, full_106, full_107]

        # pd_op.reshape_: (-1x1x2x512xf16, 0x-1x1024x1x1xf16) <- (-1x1024x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf16) <- (-1x1x2x512xf16)
        transpose_13 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf16) <- (-1x2x1x512xf16)
        softmax__13 = paddle._C_ops.softmax_(transpose_13, 1)

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_67, full_108, full_109, full_110]

        # pd_op.reshape_: (-1x1024x1x1xf16, 0x-1x2x1x512xf16) <- (-1x2x1x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__13, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf16, -1x512x1x1xf16]) <- (-1x1024x1x1xf16, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(reshape__54, 2, full_111)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_68 = split_with_num_27[0]

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__26 = paddle._C_ops.multiply_(slice_65, slice_68)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_69 = split_with_num_27[1]

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__27 = paddle._C_ops.multiply_(slice_66, slice_69)

        # pd_op.cast: (-1x512x14x14xf32) <- (-1x512x14x14xf16)
        cast_82 = paddle._C_ops.cast(multiply__26, paddle.float32)

        # pd_op.cast: (-1x512x14x14xf32) <- (-1x512x14x14xf16)
        cast_83 = paddle._C_ops.cast(multiply__27, paddle.float32)

        # builtin.combine: ([-1x512x14x14xf32, -1x512x14x14xf32]) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        combine_55 = [cast_82, cast_83]

        # pd_op.add_n: (-1x512x14x14xf32) <- ([-1x512x14x14xf32, -1x512x14x14xf32])
        add_n_27 = paddle._C_ops.add_n(combine_55)

        # pd_op.cast: (-1x512x14x14xf16) <- (-1x512x14x14xf32)
        cast_84 = paddle._C_ops.cast(add_n_27, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_62 = [3, 3]

        # pd_op.pool2d: (-1x512x7x7xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_20 = paddle._C_ops.pool2d(cast_84, full_int_array_62, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x512x7x7xf16, 2048x512x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(pool2d_20, parameter_333, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_334, parameter_335, parameter_336, parameter_337, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_63 = [2, 2]

        # pd_op.pool2d: (-1x1024x7x7xf16) <- (-1x1024x14x14xf16, 2xi64)
        pool2d_21 = paddle._C_ops.pool2d(relu__54, full_int_array_63, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x1024x7x7xf16, 2048x1024x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(pool2d_21, parameter_338, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_339, parameter_340, parameter_341, parameter_342, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__27 = paddle._C_ops.add_(batch_norm__372, batch_norm__366)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__58 = paddle._C_ops.relu_(add__27)

        # pd_op.conv2d: (-1x512x7x7xf16) <- (-1x2048x7x7xf16, 512x2048x1x1xf16)
        conv2d_77 = paddle._C_ops.conv2d(relu__58, parameter_343, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x7x7xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x7x7xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_344, parameter_345, parameter_346, parameter_347, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf16) <- (-1x512x7x7xf16)
        relu__59 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x512x7x7xf16, 1024x256x3x3xf16)
        conv2d_78 = paddle._C_ops.conv2d(relu__59, parameter_348, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_349, parameter_350, parameter_351, parameter_352, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x7x7xf16, -1x512x7x7xf16]) <- (-1x1024x7x7xf16, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(relu__60, 2, full_112)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x512x7x7xf16])
        slice_70 = split_with_num_28[0]

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_85 = paddle._C_ops.cast(slice_70, paddle.float32)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x512x7x7xf16])
        slice_71 = split_with_num_28[1]

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_86 = paddle._C_ops.cast(slice_71, paddle.float32)

        # builtin.combine: ([-1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        combine_56 = [cast_85, cast_86]

        # pd_op.add_n: (-1x512x7x7xf32) <- ([-1x512x7x7xf32, -1x512x7x7xf32])
        add_n_28 = paddle._C_ops.add_n(combine_56)

        # pd_op.cast: (-1x512x7x7xf16) <- (-1x512x7x7xf32)
        cast_87 = paddle._C_ops.cast(add_n_28, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_64 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x7x7xf16, 2xi64)
        pool2d_22 = paddle._C_ops.pool2d(cast_87, full_int_array_64, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x512x1x1xf16, 256x512x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(pool2d_22, parameter_353, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_354, parameter_355, parameter_356, parameter_357, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf16) <- (-1x256x1x1xf16)
        relu__61 = paddle._C_ops.relu_(batch_norm__390)

        # pd_op.conv2d: (-1x1024x1x1xf16) <- (-1x256x1x1xf16, 1024x256x1x1xf16)
        conv2d_80 = paddle._C_ops.conv2d(relu__61, parameter_358, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_359, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf16) <- (-1x1024x1x1xf16, 1x1024x1x1xf16)
        add__28 = paddle._C_ops.add_(conv2d_80, reshape_28)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(add__28, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(shape_14, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_72, full_113, full_114, full_115]

        # pd_op.reshape_: (-1x1x2x512xf16, 0x-1x1024x1x1xf16) <- (-1x1024x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__28, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf16) <- (-1x1x2x512xf16)
        transpose_14 = paddle._C_ops.transpose(reshape__56, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf16) <- (-1x2x1x512xf16)
        softmax__14 = paddle._C_ops.softmax_(transpose_14, 1)

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_72, full_116, full_117, full_118]

        # pd_op.reshape_: (-1x1024x1x1xf16, 0x-1x2x1x512xf16) <- (-1x2x1x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__14, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf16, -1x512x1x1xf16]) <- (-1x1024x1x1xf16, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(reshape__58, 2, full_119)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_73 = split_with_num_29[0]

        # pd_op.multiply_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x1x1xf16)
        multiply__28 = paddle._C_ops.multiply_(slice_70, slice_73)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_74 = split_with_num_29[1]

        # pd_op.multiply_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x1x1xf16)
        multiply__29 = paddle._C_ops.multiply_(slice_71, slice_74)

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_88 = paddle._C_ops.cast(multiply__28, paddle.float32)

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_89 = paddle._C_ops.cast(multiply__29, paddle.float32)

        # builtin.combine: ([-1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        combine_59 = [cast_88, cast_89]

        # pd_op.add_n: (-1x512x7x7xf32) <- ([-1x512x7x7xf32, -1x512x7x7xf32])
        add_n_29 = paddle._C_ops.add_n(combine_59)

        # pd_op.cast: (-1x512x7x7xf16) <- (-1x512x7x7xf32)
        cast_90 = paddle._C_ops.cast(add_n_29, paddle.float16)

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x512x7x7xf16, 2048x512x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(cast_90, parameter_360, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__29 = paddle._C_ops.add_(relu__58, batch_norm__396)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__62 = paddle._C_ops.relu_(add__29)

        # pd_op.conv2d: (-1x512x7x7xf16) <- (-1x2048x7x7xf16, 512x2048x1x1xf16)
        conv2d_82 = paddle._C_ops.conv2d(relu__62, parameter_365, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x7x7xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x7x7xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf16) <- (-1x512x7x7xf16)
        relu__63 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x512x7x7xf16, 1024x256x3x3xf16)
        conv2d_83 = paddle._C_ops.conv2d(relu__63, parameter_370, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__64 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x7x7xf16, -1x512x7x7xf16]) <- (-1x1024x7x7xf16, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(relu__64, 2, full_120)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x512x7x7xf16])
        slice_75 = split_with_num_30[0]

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_91 = paddle._C_ops.cast(slice_75, paddle.float32)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x512x7x7xf16])
        slice_76 = split_with_num_30[1]

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_92 = paddle._C_ops.cast(slice_76, paddle.float32)

        # builtin.combine: ([-1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        combine_60 = [cast_91, cast_92]

        # pd_op.add_n: (-1x512x7x7xf32) <- ([-1x512x7x7xf32, -1x512x7x7xf32])
        add_n_30 = paddle._C_ops.add_n(combine_60)

        # pd_op.cast: (-1x512x7x7xf16) <- (-1x512x7x7xf32)
        cast_93 = paddle._C_ops.cast(add_n_30, paddle.float16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_68 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x7x7xf16, 2xi64)
        pool2d_23 = paddle._C_ops.pool2d(cast_93, full_int_array_68, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x512x1x1xf16, 256x512x1x1xf16)
        conv2d_84 = paddle._C_ops.conv2d(pool2d_23, parameter_375, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_376, parameter_377, parameter_378, parameter_379, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf16) <- (-1x256x1x1xf16)
        relu__65 = paddle._C_ops.relu_(batch_norm__414)

        # pd_op.conv2d: (-1x1024x1x1xf16) <- (-1x256x1x1xf16, 1024x256x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu__65, parameter_380, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_381, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf16) <- (-1x1024x1x1xf16, 1x1024x1x1xf16)
        add__30 = paddle._C_ops.add_(conv2d_85, reshape_30)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(add__30, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_15, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_77, full_121, full_122, full_123]

        # pd_op.reshape_: (-1x1x2x512xf16, 0x-1x1024x1x1xf16) <- (-1x1024x1x1xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf16) <- (-1x1x2x512xf16)
        transpose_15 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf16) <- (-1x2x1x512xf16)
        softmax__15 = paddle._C_ops.softmax_(transpose_15, 1)

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_77, full_124, full_125, full_126]

        # pd_op.reshape_: (-1x1024x1x1xf16, 0x-1x2x1x512xf16) <- (-1x2x1x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__15, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf16, -1x512x1x1xf16]) <- (-1x1024x1x1xf16, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(reshape__62, 2, full_127)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_78 = split_with_num_31[0]

        # pd_op.multiply_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x1x1xf16)
        multiply__30 = paddle._C_ops.multiply_(slice_75, slice_78)

        # builtin.slice: (-1x512x1x1xf16) <- ([-1x512x1x1xf16, -1x512x1x1xf16])
        slice_79 = split_with_num_31[1]

        # pd_op.multiply_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x1x1xf16)
        multiply__31 = paddle._C_ops.multiply_(slice_76, slice_79)

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_94 = paddle._C_ops.cast(multiply__30, paddle.float32)

        # pd_op.cast: (-1x512x7x7xf32) <- (-1x512x7x7xf16)
        cast_95 = paddle._C_ops.cast(multiply__31, paddle.float32)

        # builtin.combine: ([-1x512x7x7xf32, -1x512x7x7xf32]) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        combine_63 = [cast_94, cast_95]

        # pd_op.add_n: (-1x512x7x7xf32) <- ([-1x512x7x7xf32, -1x512x7x7xf32])
        add_n_31 = paddle._C_ops.add_n(combine_63)

        # pd_op.cast: (-1x512x7x7xf16) <- (-1x512x7x7xf32)
        cast_96 = paddle._C_ops.cast(add_n_31, paddle.float16)

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x512x7x7xf16, 2048x512x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(cast_96, parameter_382, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_383, parameter_384, parameter_385, parameter_386, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__31 = paddle._C_ops.add_(relu__62, batch_norm__420)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__66 = paddle._C_ops.relu_(add__31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_72 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf16) <- (-1x2048x7x7xf16, 2xi64)
        pool2d_24 = paddle._C_ops.pool2d(relu__66, full_int_array_72, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_73 = [-1, 2048]

        # pd_op.reshape_: (-1x2048xf16, 0x-1x2048x1x1xf16) <- (-1x2048x1x1xf16, 2xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_24, full_int_array_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x2048xf16, 2048x1000xf16)
        matmul_0 = paddle._C_ops.matmul(reshape__64, parameter_387, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__32 = paddle._C_ops.add_(matmul_0, parameter_388)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__16 = paddle._C_ops.softmax_(add__32, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_97 = paddle._C_ops.cast(softmax__16, paddle.float32)
        return cast_97



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_381, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_388, feed_0):
        return self.builtin_module_1318_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_381, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_388, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1318_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([32, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1024, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_330
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([2048, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_342
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_347
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1024, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([1024, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_381
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_382
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_386
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([2048, 1000], dtype='float16', min=0, max=0.5),
            # parameter_388
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
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float16'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_330
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_332
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float16'),
            # parameter_342
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_347
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float16'),
            # parameter_352
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_381
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_382
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_386
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float16'),
            # parameter_388
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