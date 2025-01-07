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
    return [2239][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2575_0_0(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_26, parameter_25, parameter_27, parameter_28, parameter_29, parameter_30, parameter_32, parameter_31, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_56, parameter_55, parameter_57, parameter_58, parameter_59, parameter_60, parameter_62, parameter_61, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_86, parameter_85, parameter_87, parameter_88, parameter_89, parameter_90, parameter_92, parameter_91, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_126, parameter_125, parameter_127, parameter_128, parameter_129, parameter_130, parameter_132, parameter_131, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_142, parameter_143, parameter_144, parameter_146, parameter_145, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_153, parameter_152, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_194, parameter_193, parameter_195, parameter_196, parameter_197, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_234, parameter_233, parameter_235, parameter_236, parameter_237, parameter_238, parameter_240, parameter_239, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_248, parameter_247, parameter_249, parameter_250, parameter_251, parameter_252, parameter_254, parameter_253, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_281, parameter_280, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_288, parameter_287, parameter_289, parameter_290, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_311, parameter_310, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_318, parameter_317, parameter_319, parameter_320, parameter_321, parameter_322, parameter_324, parameter_323, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, feed_0):

        # pd_op.cast: (-1x3x384x384xf16) <- (-1x3x384x384xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x128x96x96xf16) <- (-1x3x384x384xf16, 128x3x4x4xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x96x96xf16) <- (-1x128x96x96xf16, 1x128x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.flatten_: (-1x128x9216xf16, None) <- (-1x128x96x96xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(add__0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x9216x128xf16) <- (-1x128x9216xf16)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.layer_norm: (-1x9216x128xf16, -9216xf32, -9216xf32) <- (-1x9216x128xf16, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(transpose_0, parameter_2, parameter_3, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x9216x128xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(layer_norm_0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.layer_norm: (-1x9216x128xf16, -9216xf32, -9216xf32) <- (-1x9216x128xf16, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(layer_norm_0, parameter_4, parameter_5, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1, full_2]

        # pd_op.reshape_: (-1x96x96x128xf16, 0x-1x9216x128xf16) <- (-1x9216x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_3, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x96x96x128xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(reshape__0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_3, full_4, full_5, full_6, full_7]

        # pd_op.reshape_: (-1x8x12x8x12x128xf16, 0x-1x96x96x128xf16) <- (-1x96x96x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x8x12x12x128xf16) <- (-1x8x12x8x12x128xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [-1, 12, 12, 128]

        # pd_op.reshape_: (-1x12x12x128xf16, 0x-1x8x8x12x12x128xf16) <- (-1x8x8x12x12x128xf16, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [-1, 144, 128]

        # pd_op.reshape_: (-1x144x128xf16, 0x-1x12x12x128xf16) <- (-1x12x12x128xf16, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__4, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x128xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(reshape__6, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.matmul: (-1x144x384xf16) <- (-1x144x128xf16, 128x384xf16)
        matmul_0 = paddle._C_ops.matmul(reshape__6, parameter_6, False, False)

        # pd_op.add_: (-1x144x384xf16) <- (-1x144x384xf16, 384xf16)
        add__1 = paddle._C_ops.add_(matmul_0, parameter_7)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_2, full_8, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x144x3x4x32xf16, 0x-1x144x384xf16) <- (-1x144x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x144x32xf16) <- (-1x144x3x4x32xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [2]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(transpose_2, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [3]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_2, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x144x32xf16) <- (-1x4x144x32xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_3, full_12, float('0'), True)

        # pd_op.transpose: (-1x4x32x144xf16) <- (-1x4x144x32xf16)
        transpose_3 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x144x144xf16) <- (-1x4x144x32xf16, -1x4x32x144xf16)
        matmul_1 = paddle._C_ops.matmul(scale__0, transpose_3, False, False)

        # pd_op.add_: (-1x4x144x144xf16) <- (-1x4x144x144xf16, 1x4x144x144xf16)
        add__2 = paddle._C_ops.add_(matmul_1, parameter_8)

        # pd_op.softmax_: (-1x4x144x144xf16) <- (-1x4x144x144xf16)
        softmax__0 = paddle._C_ops.softmax_(add__2, -1)

        # pd_op.matmul: (-1x4x144x32xf16) <- (-1x4x144x144xf16, -1x4x144x32xf16)
        matmul_2 = paddle._C_ops.matmul(softmax__0, slice_5, False, False)

        # pd_op.transpose: (-1x144x4x32xf16) <- (-1x4x144x32xf16)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_2, full_13, full_14]

        # pd_op.reshape_: (-1x144x128xf16, 0x-1x144x4x32xf16) <- (-1x144x4x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x128xf16) <- (-1x144x128xf16, 128x128xf16)
        matmul_3 = paddle._C_ops.matmul(reshape__10, parameter_9, False, False)

        # pd_op.add_: (-1x144x128xf16) <- (-1x144x128xf16, 128xf16)
        add__3 = paddle._C_ops.add_(matmul_3, parameter_10)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [-1, 12, 12, 128]

        # pd_op.reshape_: (-1x12x12x128xf16, 0x-1x144x128xf16) <- (-1x144x128xf16, 4xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__3, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_16 = [-1, 8, 8, 12, 12, 128]

        # pd_op.reshape_: (-1x8x8x12x12x128xf16, 0x-1x12x12x128xf16) <- (-1x12x12x128xf16, 6xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__12, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x12x8x12x128xf16) <- (-1x8x8x12x12x128xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__14, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [-1, 96, 96, 128]

        # pd_op.reshape_: (-1x96x96x128xf16, 0x-1x8x12x8x12x128xf16) <- (-1x8x12x8x12x128xf16, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('9216'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_0, full_15, full_16]

        # pd_op.reshape_: (-1x9216x128xf16, 0x-1x96x96x128xf16) <- (-1x96x96x128xf16, [1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__16, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, -1x9216x128xf16)
        add__4 = paddle._C_ops.add_(layer_norm_0, reshape__18)

        # pd_op.layer_norm: (-1x9216x128xf16, -9216xf32, -9216xf32) <- (-1x9216x128xf16, 128xf32, 128xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__4, parameter_11, parameter_12, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x9216x512xf16) <- (-1x9216x128xf16, 128x512xf16)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_13, False, False)

        # pd_op.add_: (-1x9216x512xf16) <- (-1x9216x512xf16, 512xf16)
        add__5 = paddle._C_ops.add_(matmul_4, parameter_14)

        # pd_op.gelu: (-1x9216x512xf16) <- (-1x9216x512xf16)
        gelu_0 = paddle._C_ops.gelu(add__5, False)

        # pd_op.matmul: (-1x9216x128xf16) <- (-1x9216x512xf16, 512x128xf16)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_15, False, False)

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, 128xf16)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_16)

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, -1x9216x128xf16)
        add__7 = paddle._C_ops.add_(add__4, add__6)

        # pd_op.shape: (3xi32) <- (-1x9216x128xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(add__7, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.layer_norm: (-1x9216x128xf16, -9216xf32, -9216xf32) <- (-1x9216x128xf16, 128xf32, 128xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__7, parameter_17, parameter_18, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_6, full_17, full_18, full_19]

        # pd_op.reshape_: (-1x96x96x128xf16, 0x-1x9216x128xf16) <- (-1x9216x128xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_9, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [-6, -6]

        # pd_op.roll: (-1x96x96x128xf16) <- (-1x96x96x128xf16, 2xi64)
        roll_0 = paddle._C_ops.roll(reshape__20, full_int_array_20, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x96x96x128xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(roll_0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_4, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_7, full_20, full_21, full_22, full_23, full_24]

        # pd_op.reshape_: (-1x8x12x8x12x128xf16, 0x-1x96x96x128xf16) <- (-1x96x96x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_0, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x8x12x12x128xf16) <- (-1x8x12x8x12x128xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__22, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [-1, 12, 12, 128]

        # pd_op.reshape_: (-1x12x12x128xf16, 0x-1x8x8x12x12x128xf16) <- (-1x8x8x12x12x128xf16, 4xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_24 = [-1, 144, 128]

        # pd_op.reshape_: (-1x144x128xf16, 0x-1x12x12x128xf16) <- (-1x12x12x128xf16, 3xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__24, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x128xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(reshape__26, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_5, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.matmul: (-1x144x384xf16) <- (-1x144x128xf16, 128x384xf16)
        matmul_6 = paddle._C_ops.matmul(reshape__26, parameter_19, False, False)

        # pd_op.add_: (-1x144x384xf16) <- (-1x144x384xf16, 384xf16)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_20)

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_8, full_25, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x144x3x4x32xf16, 0x-1x144x384xf16) <- (-1x144x384xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x4x144x32xf16) <- (-1x144x3x4x32xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [1]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_7, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_7, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [3]

        # pd_op.slice: (-1x4x144x32xf16) <- (3x-1x4x144x32xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_7, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x144x32xf16) <- (-1x4x144x32xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(slice_9, full_29, float('0'), True)

        # pd_op.transpose: (-1x4x32x144xf16) <- (-1x4x144x32xf16)
        transpose_8 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x144x144xf16) <- (-1x4x144x32xf16, -1x4x32x144xf16)
        matmul_7 = paddle._C_ops.matmul(scale__1, transpose_8, False, False)

        # pd_op.add_: (-1x4x144x144xf16) <- (-1x4x144x144xf16, 1x4x144x144xf16)
        add__9 = paddle._C_ops.add_(matmul_7, parameter_21)

        # pd_op.full: (xi32) <- ()
        full_30 = paddle._C_ops.full([], float('64'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(slice_8, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__0 = paddle._C_ops.floor_divide_(memcpy_h2d_0, full_30)

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [floor_divide__0, full_31, full_32, full_33, full_34]

        # pd_op.reshape_: (-1x64x4x144x144xf16, 0x-1x4x144x144xf16) <- (-1x4x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__9, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.unsqueeze: (64x1x144x144xf16, None) <- (64x144x144xf16, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_22, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.unsqueeze_: (1x64x1x144x144xf16, None) <- (64x1x144x144xf16, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_0, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x4x144x144xf16) <- (-1x64x4x144x144xf16, 1x64x1x144x144xf16)
        add__10 = paddle._C_ops.add_(reshape__30, unsqueeze__0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [-1, 4, 144, 144]

        # pd_op.reshape_: (-1x4x144x144xf16, 0x-1x64x4x144x144xf16) <- (-1x64x4x144x144xf16, 4xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x4x144x144xf16) <- (-1x4x144x144xf16)
        softmax__1 = paddle._C_ops.softmax_(reshape__32, -1)

        # pd_op.matmul: (-1x4x144x32xf16) <- (-1x4x144x144xf16, -1x4x144x32xf16)
        matmul_8 = paddle._C_ops.matmul(softmax__1, slice_11, False, False)

        # pd_op.transpose: (-1x144x4x32xf16) <- (-1x4x144x32xf16)
        transpose_9 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_9 = [slice_8, full_35, full_36]

        # pd_op.reshape_: (-1x144x128xf16, 0x-1x144x4x32xf16) <- (-1x144x4x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x128xf16) <- (-1x144x128xf16, 128x128xf16)
        matmul_9 = paddle._C_ops.matmul(reshape__34, parameter_23, False, False)

        # pd_op.add_: (-1x144x128xf16) <- (-1x144x128xf16, 128xf16)
        add__11 = paddle._C_ops.add_(matmul_9, parameter_24)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [-1, 12, 12, 128]

        # pd_op.reshape_: (-1x12x12x128xf16, 0x-1x144x128xf16) <- (-1x144x128xf16, 4xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_37 = [-1, 8, 8, 12, 12, 128]

        # pd_op.reshape_: (-1x8x8x12x12x128xf16, 0x-1x12x12x128xf16) <- (-1x12x12x128xf16, 6xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__36, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x12x8x12x128xf16) <- (-1x8x8x12x12x128xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__38, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [-1, 96, 96, 128]

        # pd_op.reshape_: (-1x96x96x128xf16, 0x-1x8x12x8x12x128xf16) <- (-1x8x12x8x12x128xf16, 4xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [6, 6]

        # pd_op.roll: (-1x96x96x128xf16) <- (-1x96x96x128xf16, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape__40, full_int_array_39, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('9216'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_6, full_37, full_38]

        # pd_op.reshape_: (-1x9216x128xf16, 0x-1x96x96x128xf16) <- (-1x96x96x128xf16, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_1, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, -1x9216x128xf16)
        add__12 = paddle._C_ops.add_(add__7, reshape__42)

        # pd_op.layer_norm: (-1x9216x128xf16, -9216xf32, -9216xf32) <- (-1x9216x128xf16, 128xf32, 128xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__12, parameter_25, parameter_26, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x9216x512xf16) <- (-1x9216x128xf16, 128x512xf16)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_27, False, False)

        # pd_op.add_: (-1x9216x512xf16) <- (-1x9216x512xf16, 512xf16)
        add__13 = paddle._C_ops.add_(matmul_10, parameter_28)

        # pd_op.gelu: (-1x9216x512xf16) <- (-1x9216x512xf16)
        gelu_1 = paddle._C_ops.gelu(add__13, False)

        # pd_op.matmul: (-1x9216x128xf16) <- (-1x9216x512xf16, 512x128xf16)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_29, False, False)

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, 128xf16)
        add__14 = paddle._C_ops.add_(matmul_11, parameter_30)

        # pd_op.add_: (-1x9216x128xf16) <- (-1x9216x128xf16, -1x9216x128xf16)
        add__15 = paddle._C_ops.add_(add__12, add__14)

        # pd_op.shape: (3xi32) <- (-1x9216x128xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(add__15, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_6, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_12, full_39, full_40, full_41, full_42, full_43]

        # pd_op.reshape_: (-1x48x2x48x2x128xf16, 0x-1x9216x128xf16) <- (-1x9216x128xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__15, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x48x48x2x2x128xf16) <- (-1x48x2x48x2x128xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__44, [0, 1, 3, 4, 2, 5])

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('2304'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_12, full_44, full_45]

        # pd_op.reshape_: (-1x2304x512xf16, 0x-1x48x48x2x2x128xf16) <- (-1x48x48x2x2x128xf16, [1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x2304x512xf16, -2304xf32, -2304xf32) <- (-1x2304x512xf16, 512xf32, 512xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__46, parameter_31, parameter_32, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x2304x256xf16) <- (-1x2304x512xf16, 512x256xf16)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_33, False, False)

        # pd_op.shape: (3xi32) <- (-1x2304x256xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(matmul_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_7, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.layer_norm: (-1x2304x256xf16, -2304xf32, -2304xf32) <- (-1x2304x256xf16, 256xf32, 256xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_12, parameter_34, parameter_35, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_13, full_46, full_47, full_48]

        # pd_op.reshape_: (-1x48x48x256xf16, 0x-1x2304x256xf16) <- (-1x2304x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_18, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x48x48x256xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(reshape__48, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_8, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_14, full_49, full_50, full_51, full_52, full_53]

        # pd_op.reshape_: (-1x4x12x4x12x256xf16, 0x-1x48x48x256xf16) <- (-1x48x48x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__48, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x4x12x12x256xf16) <- (-1x4x12x4x12x256xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__50, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [-1, 12, 12, 256]

        # pd_op.reshape_: (-1x12x12x256xf16, 0x-1x4x4x12x12x256xf16) <- (-1x4x4x12x12x256xf16, 4xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_47 = [-1, 144, 256]

        # pd_op.reshape_: (-1x144x256xf16, 0x-1x12x12x256xf16) <- (-1x12x12x256xf16, 3xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__52, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x256xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(reshape__54, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_9, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.matmul: (-1x144x768xf16) <- (-1x144x256xf16, 256x768xf16)
        matmul_13 = paddle._C_ops.matmul(reshape__54, parameter_36, False, False)

        # pd_op.add_: (-1x144x768xf16) <- (-1x144x768xf16, 768xf16)
        add__16 = paddle._C_ops.add_(matmul_13, parameter_37)

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_15, full_54, full_55, full_56, full_57]

        # pd_op.reshape_: (-1x144x3x8x32xf16, 0x-1x144x768xf16) <- (-1x144x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__16, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x144x32xf16) <- (-1x144x3x8x32xf16)
        transpose_13 = paddle._C_ops.transpose(reshape__56, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [1]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_13, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [2]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_13, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [3]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_13, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x144x32xf16) <- (-1x8x144x32xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(slice_16, full_58, float('0'), True)

        # pd_op.transpose: (-1x8x32x144xf16) <- (-1x8x144x32xf16)
        transpose_14 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x144x144xf16) <- (-1x8x144x32xf16, -1x8x32x144xf16)
        matmul_14 = paddle._C_ops.matmul(scale__2, transpose_14, False, False)

        # pd_op.add_: (-1x8x144x144xf16) <- (-1x8x144x144xf16, 1x8x144x144xf16)
        add__17 = paddle._C_ops.add_(matmul_14, parameter_38)

        # pd_op.softmax_: (-1x8x144x144xf16) <- (-1x8x144x144xf16)
        softmax__2 = paddle._C_ops.softmax_(add__17, -1)

        # pd_op.matmul: (-1x8x144x32xf16) <- (-1x8x144x144xf16, -1x8x144x32xf16)
        matmul_15 = paddle._C_ops.matmul(softmax__2, slice_18, False, False)

        # pd_op.transpose: (-1x144x8x32xf16) <- (-1x8x144x32xf16)
        transpose_15 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_16 = [slice_15, full_59, full_60]

        # pd_op.reshape_: (-1x144x256xf16, 0x-1x144x8x32xf16) <- (-1x144x8x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x256xf16) <- (-1x144x256xf16, 256x256xf16)
        matmul_16 = paddle._C_ops.matmul(reshape__58, parameter_39, False, False)

        # pd_op.add_: (-1x144x256xf16) <- (-1x144x256xf16, 256xf16)
        add__18 = paddle._C_ops.add_(matmul_16, parameter_40)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [-1, 12, 12, 256]

        # pd_op.reshape_: (-1x12x12x256xf16, 0x-1x144x256xf16) <- (-1x144x256xf16, 4xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_57 = [-1, 4, 4, 12, 12, 256]

        # pd_op.reshape_: (-1x4x4x12x12x256xf16, 0x-1x12x12x256xf16) <- (-1x12x12x256xf16, 6xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__60, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x12x4x12x256xf16) <- (-1x4x4x12x12x256xf16)
        transpose_16 = paddle._C_ops.transpose(reshape__62, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [-1, 48, 48, 256]

        # pd_op.reshape_: (-1x48x48x256xf16, 0x-1x4x12x4x12x256xf16) <- (-1x4x12x4x12x256xf16, 4xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('2304'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_13, full_61, full_62]

        # pd_op.reshape_: (-1x2304x256xf16, 0x-1x48x48x256xf16) <- (-1x48x48x256xf16, [1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__64, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, -1x2304x256xf16)
        add__19 = paddle._C_ops.add_(matmul_12, reshape__66)

        # pd_op.layer_norm: (-1x2304x256xf16, -2304xf32, -2304xf32) <- (-1x2304x256xf16, 256xf32, 256xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_41, parameter_42, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x2304x1024xf16) <- (-1x2304x256xf16, 256x1024xf16)
        matmul_17 = paddle._C_ops.matmul(layer_norm_21, parameter_43, False, False)

        # pd_op.add_: (-1x2304x1024xf16) <- (-1x2304x1024xf16, 1024xf16)
        add__20 = paddle._C_ops.add_(matmul_17, parameter_44)

        # pd_op.gelu: (-1x2304x1024xf16) <- (-1x2304x1024xf16)
        gelu_2 = paddle._C_ops.gelu(add__20, False)

        # pd_op.matmul: (-1x2304x256xf16) <- (-1x2304x1024xf16, 1024x256xf16)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_45, False, False)

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, 256xf16)
        add__21 = paddle._C_ops.add_(matmul_18, parameter_46)

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, -1x2304x256xf16)
        add__22 = paddle._C_ops.add_(add__19, add__21)

        # pd_op.shape: (3xi32) <- (-1x2304x256xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(add__22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_10, [0], full_int_array_59, full_int_array_60, [1], [0])

        # pd_op.layer_norm: (-1x2304x256xf16, -2304xf32, -2304xf32) <- (-1x2304x256xf16, 256xf32, 256xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_47, parameter_48, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_19, full_63, full_64, full_65]

        # pd_op.reshape_: (-1x48x48x256xf16, 0x-1x2304x256xf16) <- (-1x2304x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_24, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_61 = [-6, -6]

        # pd_op.roll: (-1x48x48x256xf16) <- (-1x48x48x256xf16, 2xi64)
        roll_2 = paddle._C_ops.roll(reshape__68, full_int_array_61, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x48x48x256xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(roll_2, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_11, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_20, full_66, full_67, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x4x12x4x12x256xf16, 0x-1x48x48x256xf16) <- (-1x48x48x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_2, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x4x12x12x256xf16) <- (-1x4x12x4x12x256xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__70, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_64 = [-1, 12, 12, 256]

        # pd_op.reshape_: (-1x12x12x256xf16, 0x-1x4x4x12x12x256xf16) <- (-1x4x4x12x12x256xf16, 4xi64)
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_65 = [-1, 144, 256]

        # pd_op.reshape_: (-1x144x256xf16, 0x-1x12x12x256xf16) <- (-1x12x12x256xf16, 3xi64)
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__72, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x256xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(reshape__74, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_12, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.matmul: (-1x144x768xf16) <- (-1x144x256xf16, 256x768xf16)
        matmul_19 = paddle._C_ops.matmul(reshape__74, parameter_49, False, False)

        # pd_op.add_: (-1x144x768xf16) <- (-1x144x768xf16, 768xf16)
        add__23 = paddle._C_ops.add_(matmul_19, parameter_50)

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_21, full_71, full_72, full_73, full_74]

        # pd_op.reshape_: (-1x144x3x8x32xf16, 0x-1x144x768xf16) <- (-1x144x768xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__23, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x8x144x32xf16) <- (-1x144x3x8x32xf16)
        transpose_18 = paddle._C_ops.transpose(reshape__76, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_18, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [2]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_18, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [3]

        # pd_op.slice: (-1x8x144x32xf16) <- (3x-1x8x144x32xf16, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_18, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_75 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x144x32xf16) <- (-1x8x144x32xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(slice_22, full_75, float('0'), True)

        # pd_op.transpose: (-1x8x32x144xf16) <- (-1x8x144x32xf16)
        transpose_19 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x144x144xf16) <- (-1x8x144x32xf16, -1x8x32x144xf16)
        matmul_20 = paddle._C_ops.matmul(scale__3, transpose_19, False, False)

        # pd_op.add_: (-1x8x144x144xf16) <- (-1x8x144x144xf16, 1x8x144x144xf16)
        add__24 = paddle._C_ops.add_(matmul_20, parameter_51)

        # pd_op.full: (xi32) <- ()
        full_76 = paddle._C_ops.full([], float('16'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(slice_21, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__1 = paddle._C_ops.floor_divide_(memcpy_h2d_1, full_76)

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [floor_divide__1, full_77, full_78, full_79, full_80]

        # pd_op.reshape_: (-1x16x8x144x144xf16, 0x-1x8x144x144xf16) <- (-1x8x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__24, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.unsqueeze: (16x1x144x144xf16, None) <- (16x144x144xf16, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_52, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [0]

        # pd_op.unsqueeze_: (1x16x1x144x144xf16, None) <- (16x1x144x144xf16, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_2, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x8x144x144xf16) <- (-1x16x8x144x144xf16, 1x16x1x144x144xf16)
        add__25 = paddle._C_ops.add_(reshape__78, unsqueeze__2)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_76 = [-1, 8, 144, 144]

        # pd_op.reshape_: (-1x8x144x144xf16, 0x-1x16x8x144x144xf16) <- (-1x16x8x144x144xf16, 4xi64)
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__25, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x8x144x144xf16) <- (-1x8x144x144xf16)
        softmax__3 = paddle._C_ops.softmax_(reshape__80, -1)

        # pd_op.matmul: (-1x8x144x32xf16) <- (-1x8x144x144xf16, -1x8x144x32xf16)
        matmul_21 = paddle._C_ops.matmul(softmax__3, slice_24, False, False)

        # pd_op.transpose: (-1x144x8x32xf16) <- (-1x8x144x32xf16)
        transpose_20 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_21, full_81, full_82]

        # pd_op.reshape_: (-1x144x256xf16, 0x-1x144x8x32xf16) <- (-1x144x8x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x256xf16) <- (-1x144x256xf16, 256x256xf16)
        matmul_22 = paddle._C_ops.matmul(reshape__82, parameter_53, False, False)

        # pd_op.add_: (-1x144x256xf16) <- (-1x144x256xf16, 256xf16)
        add__26 = paddle._C_ops.add_(matmul_22, parameter_54)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_77 = [-1, 12, 12, 256]

        # pd_op.reshape_: (-1x12x12x256xf16, 0x-1x144x256xf16) <- (-1x144x256xf16, 4xi64)
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_78 = [-1, 4, 4, 12, 12, 256]

        # pd_op.reshape_: (-1x4x4x12x12x256xf16, 0x-1x12x12x256xf16) <- (-1x12x12x256xf16, 6xi64)
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__84, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x12x4x12x256xf16) <- (-1x4x4x12x12x256xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__86, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [-1, 48, 48, 256]

        # pd_op.reshape_: (-1x48x48x256xf16, 0x-1x4x12x4x12x256xf16) <- (-1x4x12x4x12x256xf16, 4xi64)
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_80 = [6, 6]

        # pd_op.roll: (-1x48x48x256xf16) <- (-1x48x48x256xf16, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape__88, full_int_array_80, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('2304'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_23 = [slice_19, full_83, full_84]

        # pd_op.reshape_: (-1x2304x256xf16, 0x-1x48x48x256xf16) <- (-1x48x48x256xf16, [1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_3, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, -1x2304x256xf16)
        add__27 = paddle._C_ops.add_(add__22, reshape__90)

        # pd_op.layer_norm: (-1x2304x256xf16, -2304xf32, -2304xf32) <- (-1x2304x256xf16, 256xf32, 256xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__27, parameter_55, parameter_56, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x2304x1024xf16) <- (-1x2304x256xf16, 256x1024xf16)
        matmul_23 = paddle._C_ops.matmul(layer_norm_27, parameter_57, False, False)

        # pd_op.add_: (-1x2304x1024xf16) <- (-1x2304x1024xf16, 1024xf16)
        add__28 = paddle._C_ops.add_(matmul_23, parameter_58)

        # pd_op.gelu: (-1x2304x1024xf16) <- (-1x2304x1024xf16)
        gelu_3 = paddle._C_ops.gelu(add__28, False)

        # pd_op.matmul: (-1x2304x256xf16) <- (-1x2304x1024xf16, 1024x256xf16)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_59, False, False)

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, 256xf16)
        add__29 = paddle._C_ops.add_(matmul_24, parameter_60)

        # pd_op.add_: (-1x2304x256xf16) <- (-1x2304x256xf16, -1x2304x256xf16)
        add__30 = paddle._C_ops.add_(add__27, add__29)

        # pd_op.shape: (3xi32) <- (-1x2304x256xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(add__30, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_13, [0], full_int_array_81, full_int_array_82, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_25, full_85, full_86, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x24x2x24x2x256xf16, 0x-1x2304x256xf16) <- (-1x2304x256xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x24x24x2x2x256xf16) <- (-1x24x2x24x2x256xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__92, [0, 1, 3, 4, 2, 5])

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_25, full_90, full_91]

        # pd_op.reshape_: (-1x576x1024xf16, 0x-1x24x24x2x2x256xf16) <- (-1x24x24x2x2x256xf16, [1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x576x1024xf16, -576xf32, -576xf32) <- (-1x576x1024xf16, 1024xf32, 1024xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__94, parameter_61, parameter_62, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x1024xf16, 1024x512xf16)
        matmul_25 = paddle._C_ops.matmul(layer_norm_30, parameter_63, False, False)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(matmul_25, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_14, [0], full_int_array_83, full_int_array_84, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_25, parameter_64, parameter_65, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_26, full_92, full_93, full_94]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_33, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(reshape__96, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_15, [0], full_int_array_85, full_int_array_86, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_27, full_95, full_96, full_97, full_98, full_99]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__96, [x.reshape([]) for x in combine_27]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_23 = paddle._C_ops.transpose(reshape__98, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_88 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__100, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(reshape__102, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_16, [0], full_int_array_89, full_int_array_90, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_26 = paddle._C_ops.matmul(reshape__102, parameter_66, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__31 = paddle._C_ops.add_(matmul_26, parameter_67)

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_28, full_100, full_101, full_102, full_103]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__31, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_24 = paddle._C_ops.transpose(reshape__104, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_24, [0], full_int_array_91, full_int_array_92, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_24, [0], full_int_array_93, full_int_array_94, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_24, [0], full_int_array_95, full_int_array_96, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_104 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(slice_29, full_104, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_25 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_27 = paddle._C_ops.matmul(scale__4, transpose_25, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__32 = paddle._C_ops.add_(matmul_27, parameter_68)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__4 = paddle._C_ops.softmax_(add__32, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_28 = paddle._C_ops.matmul(softmax__4, slice_31, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_26 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_29 = [slice_28, full_105, full_106]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_29 = paddle._C_ops.matmul(reshape__106, parameter_69, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__33 = paddle._C_ops.add_(matmul_29, parameter_70)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_97 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__33, full_int_array_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_98 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__108, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__110, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_26, full_107, full_108]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__112, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__34 = paddle._C_ops.add_(matmul_25, reshape__114)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__34, parameter_71, parameter_72, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_30 = paddle._C_ops.matmul(layer_norm_36, parameter_73, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__35 = paddle._C_ops.add_(matmul_30, parameter_74)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_4 = paddle._C_ops.gelu(add__35, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_31 = paddle._C_ops.matmul(gelu_4, parameter_75, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__36 = paddle._C_ops.add_(matmul_31, parameter_76)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__37 = paddle._C_ops.add_(add__34, add__36)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(add__37, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_17, [0], full_int_array_100, full_int_array_101, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_77, parameter_78, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_32, full_109, full_110, full_111]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_39, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_102 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_4 = paddle._C_ops.roll(reshape__116, full_int_array_102, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(roll_4, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(shape_18, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_33, full_112, full_113, full_114, full_115, full_116]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_4, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__118, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_105 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, full_int_array_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_106 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__120, full_int_array_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(reshape__122, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(shape_19, [0], full_int_array_107, full_int_array_108, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_32 = paddle._C_ops.matmul(reshape__122, parameter_79, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__38 = paddle._C_ops.add_(matmul_32, parameter_80)

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_34, full_117, full_118, full_119, full_120]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__124, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_29, [0], full_int_array_109, full_int_array_110, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(transpose_29, [0], full_int_array_111, full_int_array_112, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(transpose_29, [0], full_int_array_113, full_int_array_114, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_121 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_35, full_121, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_30 = paddle._C_ops.transpose(slice_36, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_33 = paddle._C_ops.matmul(scale__5, transpose_30, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__39 = paddle._C_ops.add_(matmul_33, parameter_81)

        # pd_op.full: (xi32) <- ()
        full_122 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(slice_34, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__2 = paddle._C_ops.floor_divide_(memcpy_h2d_2, full_122)

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [floor_divide__2, full_123, full_124, full_125, full_126]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_82, full_int_array_115), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_4, full_int_array_116), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__40 = paddle._C_ops.add_(reshape__126, unsqueeze__4)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_117 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__40, full_int_array_117), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__5 = paddle._C_ops.softmax_(reshape__128, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_34 = paddle._C_ops.matmul(softmax__5, slice_37, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_31 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_35 = [slice_34, full_127, full_128]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_31, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_35 = paddle._C_ops.matmul(reshape__130, parameter_83, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__41 = paddle._C_ops.add_(matmul_35, parameter_84)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_118 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__41, full_int_array_118), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_119 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__132, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__134, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_120 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_32, full_int_array_120), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_121 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape__136, full_int_array_121, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_32, full_129, full_130]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_5, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__42 = paddle._C_ops.add_(add__37, reshape__138)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__42, parameter_85, parameter_86, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_36 = paddle._C_ops.matmul(layer_norm_42, parameter_87, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__43 = paddle._C_ops.add_(matmul_36, parameter_88)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_5 = paddle._C_ops.gelu(add__43, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_37 = paddle._C_ops.matmul(gelu_5, parameter_89, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__44 = paddle._C_ops.add_(matmul_37, parameter_90)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__45 = paddle._C_ops.add_(add__42, add__44)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(add__45, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_20, [0], full_int_array_122, full_int_array_123, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__45, parameter_91, parameter_92, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_38, full_131, full_132, full_133]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_45, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(reshape__140, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(shape_21, [0], full_int_array_124, full_int_array_125, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_39, full_134, full_135, full_136, full_137, full_138]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__140, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_33 = paddle._C_ops.transpose(reshape__142, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_126 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, full_int_array_126), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_127 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__144, full_int_array_127), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(reshape__146, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(shape_22, [0], full_int_array_128, full_int_array_129, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_38 = paddle._C_ops.matmul(reshape__146, parameter_93, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__46 = paddle._C_ops.add_(matmul_38, parameter_94)

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_39 = [slice_40, full_139, full_140, full_141, full_142]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__46, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__148, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(transpose_34, [0], full_int_array_130, full_int_array_131, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_34, [0], full_int_array_132, full_int_array_133, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_34, [0], full_int_array_134, full_int_array_135, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_143 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__6 = paddle._C_ops.scale_(slice_41, full_143, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_35 = paddle._C_ops.transpose(slice_42, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_39 = paddle._C_ops.matmul(scale__6, transpose_35, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__47 = paddle._C_ops.add_(matmul_39, parameter_95)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__6 = paddle._C_ops.softmax_(add__47, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_40 = paddle._C_ops.matmul(softmax__6, slice_43, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_36 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_40, full_144, full_145]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_41 = paddle._C_ops.matmul(reshape__150, parameter_96, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__48 = paddle._C_ops.add_(matmul_41, parameter_97)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_136 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, full_int_array_136), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_137 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__152, full_int_array_137), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__154, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_138 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, full_int_array_138), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_41 = [slice_38, full_146, full_147]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__156, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__49 = paddle._C_ops.add_(add__45, reshape__158)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_98, parameter_99, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_42 = paddle._C_ops.matmul(layer_norm_48, parameter_100, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__50 = paddle._C_ops.add_(matmul_42, parameter_101)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_6 = paddle._C_ops.gelu(add__50, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_43 = paddle._C_ops.matmul(gelu_6, parameter_102, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__51 = paddle._C_ops.add_(matmul_43, parameter_103)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__52 = paddle._C_ops.add_(add__49, add__51)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(add__52, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_23, [0], full_int_array_139, full_int_array_140, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_104, parameter_105, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_44, full_148, full_149, full_150]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_51, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_141 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_6 = paddle._C_ops.roll(reshape__160, full_int_array_141, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(roll_6, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(shape_24, [0], full_int_array_142, full_int_array_143, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_45, full_151, full_152, full_153, full_154, full_155]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_6, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_38 = paddle._C_ops.transpose(reshape__162, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_144 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, full_int_array_144), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_145 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__164, full_int_array_145), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(reshape__166, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_25, [0], full_int_array_146, full_int_array_147, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_44 = paddle._C_ops.matmul(reshape__166, parameter_106, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__53 = paddle._C_ops.add_(matmul_44, parameter_107)

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_46, full_156, full_157, full_158, full_159]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__53, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_39 = paddle._C_ops.transpose(reshape__168, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_39, [0], full_int_array_148, full_int_array_149, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_39, [0], full_int_array_150, full_int_array_151, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(transpose_39, [0], full_int_array_152, full_int_array_153, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_160 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__7 = paddle._C_ops.scale_(slice_47, full_160, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_40 = paddle._C_ops.transpose(slice_48, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_45 = paddle._C_ops.matmul(scale__7, transpose_40, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__54 = paddle._C_ops.add_(matmul_45, parameter_108)

        # pd_op.full: (xi32) <- ()
        full_161 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_3 = paddle._C_ops.memcpy_h2d(slice_46, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__3 = paddle._C_ops.floor_divide_(memcpy_h2d_3, full_161)

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [floor_divide__3, full_162, full_163, full_164, full_165]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__54, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_109, full_int_array_154), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_6, full_int_array_155), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__55 = paddle._C_ops.add_(reshape__170, unsqueeze__6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_156 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__55, full_int_array_156), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__7 = paddle._C_ops.softmax_(reshape__172, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_46 = paddle._C_ops.matmul(softmax__7, slice_49, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_41 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_46, full_166, full_167]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_41, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_47 = paddle._C_ops.matmul(reshape__174, parameter_110, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__56 = paddle._C_ops.add_(matmul_47, parameter_111)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_157 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, full_int_array_157), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_158 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__176, full_int_array_158), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_42 = paddle._C_ops.transpose(reshape__178, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_159 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, full_int_array_159), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_160 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_7 = paddle._C_ops.roll(reshape__180, full_int_array_160, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_44, full_168, full_169]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_7, [x.reshape([]) for x in combine_47]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__57 = paddle._C_ops.add_(add__52, reshape__182)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__57, parameter_112, parameter_113, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_48 = paddle._C_ops.matmul(layer_norm_54, parameter_114, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__58 = paddle._C_ops.add_(matmul_48, parameter_115)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_7 = paddle._C_ops.gelu(add__58, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_49 = paddle._C_ops.matmul(gelu_7, parameter_116, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__59 = paddle._C_ops.add_(matmul_49, parameter_117)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__60 = paddle._C_ops.add_(add__57, add__59)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(add__60, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_26, [0], full_int_array_161, full_int_array_162, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__60, parameter_118, parameter_119, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_50, full_170, full_171, full_172]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_57, [x.reshape([]) for x in combine_48]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(reshape__184, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(shape_27, [0], full_int_array_163, full_int_array_164, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_51, full_173, full_174, full_175, full_176, full_177]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__184, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_43 = paddle._C_ops.transpose(reshape__186, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_165 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__188, reshape__189 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_43, full_int_array_165), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_166 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__190, reshape__191 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__188, full_int_array_166), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(reshape__190, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_28, [0], full_int_array_167, full_int_array_168, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_50 = paddle._C_ops.matmul(reshape__190, parameter_120, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__61 = paddle._C_ops.add_(matmul_50, parameter_121)

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_52, full_178, full_179, full_180, full_181]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__192, reshape__193 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__61, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_44 = paddle._C_ops.transpose(reshape__192, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(transpose_44, [0], full_int_array_169, full_int_array_170, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(transpose_44, [0], full_int_array_171, full_int_array_172, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(transpose_44, [0], full_int_array_173, full_int_array_174, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_182 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__8 = paddle._C_ops.scale_(slice_53, full_182, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_45 = paddle._C_ops.transpose(slice_54, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_51 = paddle._C_ops.matmul(scale__8, transpose_45, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__62 = paddle._C_ops.add_(matmul_51, parameter_122)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__8 = paddle._C_ops.softmax_(add__62, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_52 = paddle._C_ops.matmul(softmax__8, slice_55, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_46 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_51 = [slice_52, full_183, full_184]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__194, reshape__195 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_46, [x.reshape([]) for x in combine_51]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_53 = paddle._C_ops.matmul(reshape__194, parameter_123, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__63 = paddle._C_ops.add_(matmul_53, parameter_124)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_175 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__196, reshape__197 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__63, full_int_array_175), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_176 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__198, reshape__199 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__196, full_int_array_176), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_47 = paddle._C_ops.transpose(reshape__198, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_177 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__200, reshape__201 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_47, full_int_array_177), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_52 = [slice_50, full_185, full_186]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__202, reshape__203 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__200, [x.reshape([]) for x in combine_52]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__64 = paddle._C_ops.add_(add__60, reshape__202)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__64, parameter_125, parameter_126, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_54 = paddle._C_ops.matmul(layer_norm_60, parameter_127, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__65 = paddle._C_ops.add_(matmul_54, parameter_128)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_8 = paddle._C_ops.gelu(add__65, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_55 = paddle._C_ops.matmul(gelu_8, parameter_129, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__66 = paddle._C_ops.add_(matmul_55, parameter_130)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__67 = paddle._C_ops.add_(add__64, add__66)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_29 = paddle._C_ops.shape(paddle.cast(add__67, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_29, [0], full_int_array_178, full_int_array_179, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__67, parameter_131, parameter_132, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_56, full_187, full_188, full_189]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__204, reshape__205 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_63, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_180 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_8 = paddle._C_ops.roll(reshape__204, full_int_array_180, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_30 = paddle._C_ops.shape(paddle.cast(roll_8, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_30, [0], full_int_array_181, full_int_array_182, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_192 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_57, full_190, full_191, full_192, full_193, full_194]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__206, reshape__207 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_8, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_48 = paddle._C_ops.transpose(reshape__206, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_183 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__208, reshape__209 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_48, full_int_array_183), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_184 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__210, reshape__211 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__208, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_31 = paddle._C_ops.shape(paddle.cast(reshape__210, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(shape_31, [0], full_int_array_185, full_int_array_186, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_56 = paddle._C_ops.matmul(reshape__210, parameter_133, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__68 = paddle._C_ops.add_(matmul_56, parameter_134)

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_198 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_58, full_195, full_196, full_197, full_198]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__212, reshape__213 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__68, [x.reshape([]) for x in combine_55]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_49 = paddle._C_ops.transpose(reshape__212, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_49, [0], full_int_array_187, full_int_array_188, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(transpose_49, [0], full_int_array_189, full_int_array_190, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(transpose_49, [0], full_int_array_191, full_int_array_192, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_199 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__9 = paddle._C_ops.scale_(slice_59, full_199, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_50 = paddle._C_ops.transpose(slice_60, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_57 = paddle._C_ops.matmul(scale__9, transpose_50, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__69 = paddle._C_ops.add_(matmul_57, parameter_135)

        # pd_op.full: (xi32) <- ()
        full_200 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_4 = paddle._C_ops.memcpy_h2d(slice_58, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__4 = paddle._C_ops.floor_divide_(memcpy_h2d_4, full_200)

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [floor_divide__4, full_201, full_202, full_203, full_204]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__214, reshape__215 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__69, [x.reshape([]) for x in combine_56]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_136, full_int_array_193), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_8, full_int_array_194), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__70 = paddle._C_ops.add_(reshape__214, unsqueeze__8)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_195 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__216, reshape__217 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__70, full_int_array_195), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__9 = paddle._C_ops.softmax_(reshape__216, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_58 = paddle._C_ops.matmul(softmax__9, slice_61, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_51 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_57 = [slice_58, full_205, full_206]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__218, reshape__219 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_51, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_59 = paddle._C_ops.matmul(reshape__218, parameter_137, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__71 = paddle._C_ops.add_(matmul_59, parameter_138)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_196 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__220, reshape__221 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__71, full_int_array_196), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_197 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__222, reshape__223 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__220, full_int_array_197), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_52 = paddle._C_ops.transpose(reshape__222, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_198 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__224, reshape__225 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_52, full_int_array_198), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_199 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_9 = paddle._C_ops.roll(reshape__224, full_int_array_199, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_207 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_208 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_58 = [slice_56, full_207, full_208]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__226, reshape__227 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_9, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__72 = paddle._C_ops.add_(add__67, reshape__226)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__72, parameter_139, parameter_140, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_60 = paddle._C_ops.matmul(layer_norm_66, parameter_141, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__73 = paddle._C_ops.add_(matmul_60, parameter_142)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_9 = paddle._C_ops.gelu(add__73, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_61 = paddle._C_ops.matmul(gelu_9, parameter_143, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__74 = paddle._C_ops.add_(matmul_61, parameter_144)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__75 = paddle._C_ops.add_(add__72, add__74)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_32 = paddle._C_ops.shape(paddle.cast(add__75, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_201 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_32, [0], full_int_array_200, full_int_array_201, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__75, parameter_145, parameter_146, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_62, full_209, full_210, full_211]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__228, reshape__229 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_69, [x.reshape([]) for x in combine_59]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_33 = paddle._C_ops.shape(paddle.cast(reshape__228, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(shape_33, [0], full_int_array_202, full_int_array_203, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_60 = [slice_63, full_212, full_213, full_214, full_215, full_216]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__230, reshape__231 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__228, [x.reshape([]) for x in combine_60]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_53 = paddle._C_ops.transpose(reshape__230, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_204 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__232, reshape__233 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_53, full_int_array_204), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_205 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__234, reshape__235 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__232, full_int_array_205), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_34 = paddle._C_ops.shape(paddle.cast(reshape__234, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(shape_34, [0], full_int_array_206, full_int_array_207, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_62 = paddle._C_ops.matmul(reshape__234, parameter_147, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__76 = paddle._C_ops.add_(matmul_62, parameter_148)

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_64, full_217, full_218, full_219, full_220]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__236, reshape__237 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__76, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_54 = paddle._C_ops.transpose(reshape__236, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(transpose_54, [0], full_int_array_208, full_int_array_209, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_211 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_54, [0], full_int_array_210, full_int_array_211, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_212 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_213 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(transpose_54, [0], full_int_array_212, full_int_array_213, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_221 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__10 = paddle._C_ops.scale_(slice_65, full_221, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_55 = paddle._C_ops.transpose(slice_66, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_63 = paddle._C_ops.matmul(scale__10, transpose_55, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__77 = paddle._C_ops.add_(matmul_63, parameter_149)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__10 = paddle._C_ops.softmax_(add__77, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_64 = paddle._C_ops.matmul(softmax__10, slice_67, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_56 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_222 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_62 = [slice_64, full_222, full_223]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__238, reshape__239 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_56, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_65 = paddle._C_ops.matmul(reshape__238, parameter_150, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__78 = paddle._C_ops.add_(matmul_65, parameter_151)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_214 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__240, reshape__241 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__78, full_int_array_214), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_215 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__242, reshape__243 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__240, full_int_array_215), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_57 = paddle._C_ops.transpose(reshape__242, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_216 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__244, reshape__245 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_57, full_int_array_216), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_63 = [slice_62, full_224, full_225]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__246, reshape__247 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__244, [x.reshape([]) for x in combine_63]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__79 = paddle._C_ops.add_(add__75, reshape__246)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__79, parameter_152, parameter_153, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_66 = paddle._C_ops.matmul(layer_norm_72, parameter_154, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__80 = paddle._C_ops.add_(matmul_66, parameter_155)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_10 = paddle._C_ops.gelu(add__80, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_67 = paddle._C_ops.matmul(gelu_10, parameter_156, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__81 = paddle._C_ops.add_(matmul_67, parameter_157)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__82 = paddle._C_ops.add_(add__79, add__81)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_35 = paddle._C_ops.shape(paddle.cast(add__82, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_217 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_218 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_35, [0], full_int_array_217, full_int_array_218, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__82, parameter_158, parameter_159, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_64 = [slice_68, full_226, full_227, full_228]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__248, reshape__249 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_75, [x.reshape([]) for x in combine_64]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_219 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_10 = paddle._C_ops.roll(reshape__248, full_int_array_219, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_36 = paddle._C_ops.shape(paddle.cast(roll_10, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_220 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_221 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(shape_36, [0], full_int_array_220, full_int_array_221, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_230 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_69, full_229, full_230, full_231, full_232, full_233]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__250, reshape__251 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_10, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_58 = paddle._C_ops.transpose(reshape__250, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_222 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__252, reshape__253 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_58, full_int_array_222), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_223 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__254, reshape__255 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__252, full_int_array_223), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_37 = paddle._C_ops.shape(paddle.cast(reshape__254, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_224 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_37, [0], full_int_array_224, full_int_array_225, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_68 = paddle._C_ops.matmul(reshape__254, parameter_160, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__83 = paddle._C_ops.add_(matmul_68, parameter_161)

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_237 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_70, full_234, full_235, full_236, full_237]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__256, reshape__257 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__83, [x.reshape([]) for x in combine_66]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_59 = paddle._C_ops.transpose(reshape__256, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(transpose_59, [0], full_int_array_226, full_int_array_227, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_228 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_229 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_59, [0], full_int_array_228, full_int_array_229, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_230 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_231 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(transpose_59, [0], full_int_array_230, full_int_array_231, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_238 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__11 = paddle._C_ops.scale_(slice_71, full_238, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_60 = paddle._C_ops.transpose(slice_72, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_69 = paddle._C_ops.matmul(scale__11, transpose_60, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__84 = paddle._C_ops.add_(matmul_69, parameter_162)

        # pd_op.full: (xi32) <- ()
        full_239 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_5 = paddle._C_ops.memcpy_h2d(slice_70, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__5 = paddle._C_ops.floor_divide_(memcpy_h2d_5, full_239)

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_242 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_67 = [floor_divide__5, full_240, full_241, full_242, full_243]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__258, reshape__259 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__84, [x.reshape([]) for x in combine_67]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_232 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_10, unsqueeze_11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_163, full_int_array_232), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_233 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_10, full_int_array_233), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__85 = paddle._C_ops.add_(reshape__258, unsqueeze__10)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_234 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__260, reshape__261 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__85, full_int_array_234), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__11 = paddle._C_ops.softmax_(reshape__260, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_70 = paddle._C_ops.matmul(softmax__11, slice_73, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_61 = paddle._C_ops.transpose(matmul_70, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_70, full_244, full_245]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__262, reshape__263 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_61, [x.reshape([]) for x in combine_68]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_71 = paddle._C_ops.matmul(reshape__262, parameter_164, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__86 = paddle._C_ops.add_(matmul_71, parameter_165)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_235 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__264, reshape__265 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__86, full_int_array_235), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_236 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__266, reshape__267 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__264, full_int_array_236), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_62 = paddle._C_ops.transpose(reshape__266, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_237 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__268, reshape__269 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_62, full_int_array_237), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_238 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_11 = paddle._C_ops.roll(reshape__268, full_int_array_238, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_69 = [slice_68, full_246, full_247]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__270, reshape__271 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_11, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__87 = paddle._C_ops.add_(add__82, reshape__270)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__87, parameter_166, parameter_167, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_72 = paddle._C_ops.matmul(layer_norm_78, parameter_168, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__88 = paddle._C_ops.add_(matmul_72, parameter_169)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_11 = paddle._C_ops.gelu(add__88, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_73 = paddle._C_ops.matmul(gelu_11, parameter_170, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__89 = paddle._C_ops.add_(matmul_73, parameter_171)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__90 = paddle._C_ops.add_(add__87, add__89)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_38 = paddle._C_ops.shape(paddle.cast(add__90, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_239 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_240 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(shape_38, [0], full_int_array_239, full_int_array_240, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__90, parameter_172, parameter_173, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_74, full_248, full_249, full_250]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__272, reshape__273 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_81, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_39 = paddle._C_ops.shape(paddle.cast(reshape__272, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_241 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_242 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(shape_39, [0], full_int_array_241, full_int_array_242, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_252 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_71 = [slice_75, full_251, full_252, full_253, full_254, full_255]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__274, reshape__275 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__272, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_63 = paddle._C_ops.transpose(reshape__274, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_243 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__276, reshape__277 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_63, full_int_array_243), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_244 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__278, reshape__279 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__276, full_int_array_244), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_40 = paddle._C_ops.shape(paddle.cast(reshape__278, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_245 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_246 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_40, [0], full_int_array_245, full_int_array_246, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_74 = paddle._C_ops.matmul(reshape__278, parameter_174, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__91 = paddle._C_ops.add_(matmul_74, parameter_175)

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_76, full_256, full_257, full_258, full_259]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__280, reshape__281 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__91, [x.reshape([]) for x in combine_72]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_64 = paddle._C_ops.transpose(reshape__280, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_247 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_248 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(transpose_64, [0], full_int_array_247, full_int_array_248, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_249 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_250 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(transpose_64, [0], full_int_array_249, full_int_array_250, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_251 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_252 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(transpose_64, [0], full_int_array_251, full_int_array_252, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_260 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__12 = paddle._C_ops.scale_(slice_77, full_260, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_65 = paddle._C_ops.transpose(slice_78, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_75 = paddle._C_ops.matmul(scale__12, transpose_65, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__92 = paddle._C_ops.add_(matmul_75, parameter_176)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__12 = paddle._C_ops.softmax_(add__92, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_76 = paddle._C_ops.matmul(softmax__12, slice_79, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_66 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_73 = [slice_76, full_261, full_262]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__282, reshape__283 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_66, [x.reshape([]) for x in combine_73]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_77 = paddle._C_ops.matmul(reshape__282, parameter_177, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__93 = paddle._C_ops.add_(matmul_77, parameter_178)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_253 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__284, reshape__285 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__93, full_int_array_253), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_254 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__286, reshape__287 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__284, full_int_array_254), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_67 = paddle._C_ops.transpose(reshape__286, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_255 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__288, reshape__289 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_67, full_int_array_255), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_264 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_74, full_263, full_264]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__290, reshape__291 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__288, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__94 = paddle._C_ops.add_(add__90, reshape__290)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__94, parameter_179, parameter_180, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_78 = paddle._C_ops.matmul(layer_norm_84, parameter_181, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__95 = paddle._C_ops.add_(matmul_78, parameter_182)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_12 = paddle._C_ops.gelu(add__95, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_79 = paddle._C_ops.matmul(gelu_12, parameter_183, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__96 = paddle._C_ops.add_(matmul_79, parameter_184)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__97 = paddle._C_ops.add_(add__94, add__96)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_41 = paddle._C_ops.shape(paddle.cast(add__97, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_256 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_257 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(shape_41, [0], full_int_array_256, full_int_array_257, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__97, parameter_185, parameter_186, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_267 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_75 = [slice_80, full_265, full_266, full_267]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__292, reshape__293 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_87, [x.reshape([]) for x in combine_75]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_258 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_12 = paddle._C_ops.roll(reshape__292, full_int_array_258, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_42 = paddle._C_ops.shape(paddle.cast(roll_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_259 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_260 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_42, [0], full_int_array_259, full_int_array_260, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_268 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_269 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_270 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_271 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_272 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_81, full_268, full_269, full_270, full_271, full_272]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__294, reshape__295 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_12, [x.reshape([]) for x in combine_76]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_68 = paddle._C_ops.transpose(reshape__294, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_261 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__296, reshape__297 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_68, full_int_array_261), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_262 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__298, reshape__299 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__296, full_int_array_262), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_43 = paddle._C_ops.shape(paddle.cast(reshape__298, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_263 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_264 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(shape_43, [0], full_int_array_263, full_int_array_264, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_80 = paddle._C_ops.matmul(reshape__298, parameter_187, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__98 = paddle._C_ops.add_(matmul_80, parameter_188)

        # pd_op.full: (1xi32) <- ()
        full_273 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_274 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_275 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_276 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_82, full_273, full_274, full_275, full_276]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__300, reshape__301 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__98, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_69 = paddle._C_ops.transpose(reshape__300, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_265 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_266 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(transpose_69, [0], full_int_array_265, full_int_array_266, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_267 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_268 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(transpose_69, [0], full_int_array_267, full_int_array_268, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_269 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_270 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(transpose_69, [0], full_int_array_269, full_int_array_270, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_277 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__13 = paddle._C_ops.scale_(slice_83, full_277, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_70 = paddle._C_ops.transpose(slice_84, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_81 = paddle._C_ops.matmul(scale__13, transpose_70, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__99 = paddle._C_ops.add_(matmul_81, parameter_189)

        # pd_op.full: (xi32) <- ()
        full_278 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_6 = paddle._C_ops.memcpy_h2d(slice_82, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__6 = paddle._C_ops.floor_divide_(memcpy_h2d_6, full_278)

        # pd_op.full: (1xi32) <- ()
        full_279 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_280 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_281 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_282 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_78 = [floor_divide__6, full_279, full_280, full_281, full_282]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__302, reshape__303 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__99, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_271 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_12, unsqueeze_13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_190, full_int_array_271), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_272 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_12, full_int_array_272), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__100 = paddle._C_ops.add_(reshape__302, unsqueeze__12)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_273 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__304, reshape__305 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__100, full_int_array_273), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__13 = paddle._C_ops.softmax_(reshape__304, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_82 = paddle._C_ops.matmul(softmax__13, slice_85, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_71 = paddle._C_ops.transpose(matmul_82, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_283 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_284 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_79 = [slice_82, full_283, full_284]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__306, reshape__307 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_71, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_83 = paddle._C_ops.matmul(reshape__306, parameter_191, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__101 = paddle._C_ops.add_(matmul_83, parameter_192)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_274 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__308, reshape__309 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__101, full_int_array_274), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_275 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__310, reshape__311 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__308, full_int_array_275), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_72 = paddle._C_ops.transpose(reshape__310, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_276 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__312, reshape__313 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_72, full_int_array_276), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_277 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_13 = paddle._C_ops.roll(reshape__312, full_int_array_277, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_285 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_286 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_80 = [slice_80, full_285, full_286]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__314, reshape__315 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_13, [x.reshape([]) for x in combine_80]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__102 = paddle._C_ops.add_(add__97, reshape__314)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__102, parameter_193, parameter_194, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_84 = paddle._C_ops.matmul(layer_norm_90, parameter_195, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__103 = paddle._C_ops.add_(matmul_84, parameter_196)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_13 = paddle._C_ops.gelu(add__103, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_85 = paddle._C_ops.matmul(gelu_13, parameter_197, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__104 = paddle._C_ops.add_(matmul_85, parameter_198)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__105 = paddle._C_ops.add_(add__102, add__104)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_44 = paddle._C_ops.shape(paddle.cast(add__105, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_278 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_279 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_44, [0], full_int_array_278, full_int_array_279, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__105, parameter_199, parameter_200, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_287 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_288 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_289 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_81 = [slice_86, full_287, full_288, full_289]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__316, reshape__317 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_93, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_45 = paddle._C_ops.shape(paddle.cast(reshape__316, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_280 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_281 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(shape_45, [0], full_int_array_280, full_int_array_281, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_290 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_291 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_292 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_293 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_294 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_87, full_290, full_291, full_292, full_293, full_294]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__318, reshape__319 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__316, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_73 = paddle._C_ops.transpose(reshape__318, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_282 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__320, reshape__321 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_73, full_int_array_282), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_283 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__322, reshape__323 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__320, full_int_array_283), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_46 = paddle._C_ops.shape(paddle.cast(reshape__322, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_284 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_285 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(shape_46, [0], full_int_array_284, full_int_array_285, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_86 = paddle._C_ops.matmul(reshape__322, parameter_201, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__106 = paddle._C_ops.add_(matmul_86, parameter_202)

        # pd_op.full: (1xi32) <- ()
        full_295 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_296 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_297 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_298 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_83 = [slice_88, full_295, full_296, full_297, full_298]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__324, reshape__325 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__106, [x.reshape([]) for x in combine_83]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_74 = paddle._C_ops.transpose(reshape__324, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_286 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_287 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(transpose_74, [0], full_int_array_286, full_int_array_287, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_288 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_289 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(transpose_74, [0], full_int_array_288, full_int_array_289, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_290 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_291 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(transpose_74, [0], full_int_array_290, full_int_array_291, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_299 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__14 = paddle._C_ops.scale_(slice_89, full_299, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_75 = paddle._C_ops.transpose(slice_90, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_87 = paddle._C_ops.matmul(scale__14, transpose_75, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__107 = paddle._C_ops.add_(matmul_87, parameter_203)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__14 = paddle._C_ops.softmax_(add__107, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_88 = paddle._C_ops.matmul(softmax__14, slice_91, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_76 = paddle._C_ops.transpose(matmul_88, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_300 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_301 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_84 = [slice_88, full_300, full_301]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__326, reshape__327 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_76, [x.reshape([]) for x in combine_84]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_89 = paddle._C_ops.matmul(reshape__326, parameter_204, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__108 = paddle._C_ops.add_(matmul_89, parameter_205)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_292 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__328, reshape__329 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__108, full_int_array_292), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_293 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__330, reshape__331 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__328, full_int_array_293), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_77 = paddle._C_ops.transpose(reshape__330, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_294 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__332, reshape__333 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_77, full_int_array_294), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_302 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_303 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_85 = [slice_86, full_302, full_303]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__334, reshape__335 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__332, [x.reshape([]) for x in combine_85]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__109 = paddle._C_ops.add_(add__105, reshape__334)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__109, parameter_206, parameter_207, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_90 = paddle._C_ops.matmul(layer_norm_96, parameter_208, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__110 = paddle._C_ops.add_(matmul_90, parameter_209)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_14 = paddle._C_ops.gelu(add__110, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_91 = paddle._C_ops.matmul(gelu_14, parameter_210, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__111 = paddle._C_ops.add_(matmul_91, parameter_211)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__112 = paddle._C_ops.add_(add__109, add__111)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_47 = paddle._C_ops.shape(paddle.cast(add__112, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_295 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_296 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_47, [0], full_int_array_295, full_int_array_296, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__112, parameter_212, parameter_213, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_304 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_305 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_306 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_86 = [slice_92, full_304, full_305, full_306]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__336, reshape__337 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_99, [x.reshape([]) for x in combine_86]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_297 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_14 = paddle._C_ops.roll(reshape__336, full_int_array_297, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_48 = paddle._C_ops.shape(paddle.cast(roll_14, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_298 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_299 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(shape_48, [0], full_int_array_298, full_int_array_299, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_307 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_308 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_309 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_310 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_311 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_87 = [slice_93, full_307, full_308, full_309, full_310, full_311]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__338, reshape__339 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_14, [x.reshape([]) for x in combine_87]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_78 = paddle._C_ops.transpose(reshape__338, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_300 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__340, reshape__341 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_78, full_int_array_300), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_301 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__342, reshape__343 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__340, full_int_array_301), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_49 = paddle._C_ops.shape(paddle.cast(reshape__342, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_302 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_303 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_49, [0], full_int_array_302, full_int_array_303, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_92 = paddle._C_ops.matmul(reshape__342, parameter_214, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__113 = paddle._C_ops.add_(matmul_92, parameter_215)

        # pd_op.full: (1xi32) <- ()
        full_312 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_313 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_314 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_315 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_88 = [slice_94, full_312, full_313, full_314, full_315]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__344, reshape__345 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__113, [x.reshape([]) for x in combine_88]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_79 = paddle._C_ops.transpose(reshape__344, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_304 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_305 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(transpose_79, [0], full_int_array_304, full_int_array_305, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_306 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_307 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(transpose_79, [0], full_int_array_306, full_int_array_307, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_308 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_309 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(transpose_79, [0], full_int_array_308, full_int_array_309, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_316 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__15 = paddle._C_ops.scale_(slice_95, full_316, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_80 = paddle._C_ops.transpose(slice_96, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_93 = paddle._C_ops.matmul(scale__15, transpose_80, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__114 = paddle._C_ops.add_(matmul_93, parameter_216)

        # pd_op.full: (xi32) <- ()
        full_317 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_7 = paddle._C_ops.memcpy_h2d(slice_94, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__7 = paddle._C_ops.floor_divide_(memcpy_h2d_7, full_317)

        # pd_op.full: (1xi32) <- ()
        full_318 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_319 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_320 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_321 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_89 = [floor_divide__7, full_318, full_319, full_320, full_321]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__346, reshape__347 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__114, [x.reshape([]) for x in combine_89]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_310 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_14, unsqueeze_15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_217, full_int_array_310), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_311 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_14, full_int_array_311), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__115 = paddle._C_ops.add_(reshape__346, unsqueeze__14)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_312 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__348, reshape__349 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__115, full_int_array_312), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__15 = paddle._C_ops.softmax_(reshape__348, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_94 = paddle._C_ops.matmul(softmax__15, slice_97, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_81 = paddle._C_ops.transpose(matmul_94, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_322 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_323 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_90 = [slice_94, full_322, full_323]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__350, reshape__351 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_81, [x.reshape([]) for x in combine_90]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_95 = paddle._C_ops.matmul(reshape__350, parameter_218, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__116 = paddle._C_ops.add_(matmul_95, parameter_219)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_313 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__352, reshape__353 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__116, full_int_array_313), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_314 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__354, reshape__355 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__352, full_int_array_314), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_82 = paddle._C_ops.transpose(reshape__354, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_315 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__356, reshape__357 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_82, full_int_array_315), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_316 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_15 = paddle._C_ops.roll(reshape__356, full_int_array_316, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_324 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_325 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_91 = [slice_92, full_324, full_325]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__358, reshape__359 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_15, [x.reshape([]) for x in combine_91]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__117 = paddle._C_ops.add_(add__112, reshape__358)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__117, parameter_220, parameter_221, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_96 = paddle._C_ops.matmul(layer_norm_102, parameter_222, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__118 = paddle._C_ops.add_(matmul_96, parameter_223)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_15 = paddle._C_ops.gelu(add__118, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_97 = paddle._C_ops.matmul(gelu_15, parameter_224, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__119 = paddle._C_ops.add_(matmul_97, parameter_225)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__120 = paddle._C_ops.add_(add__117, add__119)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_50 = paddle._C_ops.shape(paddle.cast(add__120, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_317 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_318 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(shape_50, [0], full_int_array_317, full_int_array_318, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__120, parameter_226, parameter_227, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_326 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_327 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_328 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_92 = [slice_98, full_326, full_327, full_328]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__360, reshape__361 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_105, [x.reshape([]) for x in combine_92]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_51 = paddle._C_ops.shape(paddle.cast(reshape__360, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_319 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_320 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(shape_51, [0], full_int_array_319, full_int_array_320, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_329 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_330 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_331 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_332 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_333 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_93 = [slice_99, full_329, full_330, full_331, full_332, full_333]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__362, reshape__363 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__360, [x.reshape([]) for x in combine_93]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_83 = paddle._C_ops.transpose(reshape__362, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_321 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__364, reshape__365 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_83, full_int_array_321), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_322 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__366, reshape__367 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__364, full_int_array_322), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_52 = paddle._C_ops.shape(paddle.cast(reshape__366, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_323 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_324 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(shape_52, [0], full_int_array_323, full_int_array_324, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_98 = paddle._C_ops.matmul(reshape__366, parameter_228, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__121 = paddle._C_ops.add_(matmul_98, parameter_229)

        # pd_op.full: (1xi32) <- ()
        full_334 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_335 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_336 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_337 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_94 = [slice_100, full_334, full_335, full_336, full_337]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__368, reshape__369 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__121, [x.reshape([]) for x in combine_94]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_84 = paddle._C_ops.transpose(reshape__368, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_325 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_326 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(transpose_84, [0], full_int_array_325, full_int_array_326, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_327 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_328 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(transpose_84, [0], full_int_array_327, full_int_array_328, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_329 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_330 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(transpose_84, [0], full_int_array_329, full_int_array_330, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_338 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__16 = paddle._C_ops.scale_(slice_101, full_338, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_85 = paddle._C_ops.transpose(slice_102, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_99 = paddle._C_ops.matmul(scale__16, transpose_85, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__122 = paddle._C_ops.add_(matmul_99, parameter_230)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__16 = paddle._C_ops.softmax_(add__122, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_100 = paddle._C_ops.matmul(softmax__16, slice_103, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_86 = paddle._C_ops.transpose(matmul_100, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_339 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_340 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_95 = [slice_100, full_339, full_340]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__370, reshape__371 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_86, [x.reshape([]) for x in combine_95]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_101 = paddle._C_ops.matmul(reshape__370, parameter_231, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__123 = paddle._C_ops.add_(matmul_101, parameter_232)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_331 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__372, reshape__373 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__123, full_int_array_331), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_332 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__374, reshape__375 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__372, full_int_array_332), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_87 = paddle._C_ops.transpose(reshape__374, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_333 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__376, reshape__377 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_87, full_int_array_333), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_341 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_342 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_96 = [slice_98, full_341, full_342]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__378, reshape__379 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__376, [x.reshape([]) for x in combine_96]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__124 = paddle._C_ops.add_(add__120, reshape__378)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__124, parameter_233, parameter_234, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_102 = paddle._C_ops.matmul(layer_norm_108, parameter_235, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__125 = paddle._C_ops.add_(matmul_102, parameter_236)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_16 = paddle._C_ops.gelu(add__125, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_103 = paddle._C_ops.matmul(gelu_16, parameter_237, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__126 = paddle._C_ops.add_(matmul_103, parameter_238)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__127 = paddle._C_ops.add_(add__124, add__126)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_53 = paddle._C_ops.shape(paddle.cast(add__127, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_334 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_335 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_53, [0], full_int_array_334, full_int_array_335, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__127, parameter_239, parameter_240, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_343 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_344 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_345 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_97 = [slice_104, full_343, full_344, full_345]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__380, reshape__381 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_111, [x.reshape([]) for x in combine_97]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_336 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_16 = paddle._C_ops.roll(reshape__380, full_int_array_336, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_54 = paddle._C_ops.shape(paddle.cast(roll_16, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_337 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_338 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(shape_54, [0], full_int_array_337, full_int_array_338, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_346 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_347 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_348 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_349 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_350 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_98 = [slice_105, full_346, full_347, full_348, full_349, full_350]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__382, reshape__383 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_16, [x.reshape([]) for x in combine_98]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_88 = paddle._C_ops.transpose(reshape__382, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_339 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__384, reshape__385 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_88, full_int_array_339), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_340 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__386, reshape__387 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__384, full_int_array_340), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_55 = paddle._C_ops.shape(paddle.cast(reshape__386, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_341 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_342 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(shape_55, [0], full_int_array_341, full_int_array_342, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_104 = paddle._C_ops.matmul(reshape__386, parameter_241, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__128 = paddle._C_ops.add_(matmul_104, parameter_242)

        # pd_op.full: (1xi32) <- ()
        full_351 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_352 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_353 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_354 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_99 = [slice_106, full_351, full_352, full_353, full_354]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__388, reshape__389 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__128, [x.reshape([]) for x in combine_99]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_89 = paddle._C_ops.transpose(reshape__388, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_343 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_344 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(transpose_89, [0], full_int_array_343, full_int_array_344, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_345 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_346 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(transpose_89, [0], full_int_array_345, full_int_array_346, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_347 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_348 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(transpose_89, [0], full_int_array_347, full_int_array_348, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_355 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__17 = paddle._C_ops.scale_(slice_107, full_355, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_90 = paddle._C_ops.transpose(slice_108, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_105 = paddle._C_ops.matmul(scale__17, transpose_90, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__129 = paddle._C_ops.add_(matmul_105, parameter_243)

        # pd_op.full: (xi32) <- ()
        full_356 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_8 = paddle._C_ops.memcpy_h2d(slice_106, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__8 = paddle._C_ops.floor_divide_(memcpy_h2d_8, full_356)

        # pd_op.full: (1xi32) <- ()
        full_357 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_358 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_359 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_360 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_100 = [floor_divide__8, full_357, full_358, full_359, full_360]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__390, reshape__391 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__129, [x.reshape([]) for x in combine_100]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_349 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_16, unsqueeze_17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_244, full_int_array_349), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_350 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_16, full_int_array_350), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__130 = paddle._C_ops.add_(reshape__390, unsqueeze__16)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_351 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__392, reshape__393 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__130, full_int_array_351), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__17 = paddle._C_ops.softmax_(reshape__392, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_106 = paddle._C_ops.matmul(softmax__17, slice_109, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_91 = paddle._C_ops.transpose(matmul_106, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_361 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_362 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_101 = [slice_106, full_361, full_362]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__394, reshape__395 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_91, [x.reshape([]) for x in combine_101]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_107 = paddle._C_ops.matmul(reshape__394, parameter_245, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__131 = paddle._C_ops.add_(matmul_107, parameter_246)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_352 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__396, reshape__397 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__131, full_int_array_352), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_353 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__398, reshape__399 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__396, full_int_array_353), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_92 = paddle._C_ops.transpose(reshape__398, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_354 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__400, reshape__401 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_92, full_int_array_354), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_355 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_17 = paddle._C_ops.roll(reshape__400, full_int_array_355, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_363 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_364 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_102 = [slice_104, full_363, full_364]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__402, reshape__403 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_17, [x.reshape([]) for x in combine_102]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__132 = paddle._C_ops.add_(add__127, reshape__402)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__132, parameter_247, parameter_248, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_108 = paddle._C_ops.matmul(layer_norm_114, parameter_249, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__133 = paddle._C_ops.add_(matmul_108, parameter_250)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_17 = paddle._C_ops.gelu(add__133, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_109 = paddle._C_ops.matmul(gelu_17, parameter_251, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__134 = paddle._C_ops.add_(matmul_109, parameter_252)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__135 = paddle._C_ops.add_(add__132, add__134)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_56 = paddle._C_ops.shape(paddle.cast(add__135, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_356 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_357 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(shape_56, [0], full_int_array_356, full_int_array_357, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__135, parameter_253, parameter_254, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_365 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_366 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_367 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_103 = [slice_110, full_365, full_366, full_367]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__404, reshape__405 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_117, [x.reshape([]) for x in combine_103]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_57 = paddle._C_ops.shape(paddle.cast(reshape__404, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_358 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_359 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(shape_57, [0], full_int_array_358, full_int_array_359, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_368 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_369 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_370 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_371 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_372 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_104 = [slice_111, full_368, full_369, full_370, full_371, full_372]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__406, reshape__407 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__404, [x.reshape([]) for x in combine_104]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_93 = paddle._C_ops.transpose(reshape__406, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_360 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__408, reshape__409 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_93, full_int_array_360), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_361 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__410, reshape__411 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__408, full_int_array_361), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_58 = paddle._C_ops.shape(paddle.cast(reshape__410, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_362 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_363 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(shape_58, [0], full_int_array_362, full_int_array_363, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_110 = paddle._C_ops.matmul(reshape__410, parameter_255, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__136 = paddle._C_ops.add_(matmul_110, parameter_256)

        # pd_op.full: (1xi32) <- ()
        full_373 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_374 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_375 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_376 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_105 = [slice_112, full_373, full_374, full_375, full_376]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__412, reshape__413 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__136, [x.reshape([]) for x in combine_105]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_94 = paddle._C_ops.transpose(reshape__412, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_364 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_365 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(transpose_94, [0], full_int_array_364, full_int_array_365, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_366 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_367 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(transpose_94, [0], full_int_array_366, full_int_array_367, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_368 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_369 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(transpose_94, [0], full_int_array_368, full_int_array_369, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_377 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__18 = paddle._C_ops.scale_(slice_113, full_377, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_95 = paddle._C_ops.transpose(slice_114, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_111 = paddle._C_ops.matmul(scale__18, transpose_95, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__137 = paddle._C_ops.add_(matmul_111, parameter_257)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__18 = paddle._C_ops.softmax_(add__137, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_112 = paddle._C_ops.matmul(softmax__18, slice_115, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_96 = paddle._C_ops.transpose(matmul_112, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_378 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_379 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_106 = [slice_112, full_378, full_379]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__414, reshape__415 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_96, [x.reshape([]) for x in combine_106]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_113 = paddle._C_ops.matmul(reshape__414, parameter_258, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__138 = paddle._C_ops.add_(matmul_113, parameter_259)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_370 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__416, reshape__417 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__138, full_int_array_370), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_371 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__418, reshape__419 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__416, full_int_array_371), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_97 = paddle._C_ops.transpose(reshape__418, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_372 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__420, reshape__421 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_97, full_int_array_372), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_380 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_381 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_107 = [slice_110, full_380, full_381]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__422, reshape__423 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__420, [x.reshape([]) for x in combine_107]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__139 = paddle._C_ops.add_(add__135, reshape__422)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__139, parameter_260, parameter_261, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_114 = paddle._C_ops.matmul(layer_norm_120, parameter_262, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__140 = paddle._C_ops.add_(matmul_114, parameter_263)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_18 = paddle._C_ops.gelu(add__140, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_115 = paddle._C_ops.matmul(gelu_18, parameter_264, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__141 = paddle._C_ops.add_(matmul_115, parameter_265)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__142 = paddle._C_ops.add_(add__139, add__141)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_59 = paddle._C_ops.shape(paddle.cast(add__142, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_373 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_374 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(shape_59, [0], full_int_array_373, full_int_array_374, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__142, parameter_266, parameter_267, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_382 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_383 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_384 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_108 = [slice_116, full_382, full_383, full_384]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__424, reshape__425 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_123, [x.reshape([]) for x in combine_108]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_375 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_18 = paddle._C_ops.roll(reshape__424, full_int_array_375, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_60 = paddle._C_ops.shape(paddle.cast(roll_18, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_376 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_377 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(shape_60, [0], full_int_array_376, full_int_array_377, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_385 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_386 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_387 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_388 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_389 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_109 = [slice_117, full_385, full_386, full_387, full_388, full_389]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__426, reshape__427 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_18, [x.reshape([]) for x in combine_109]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_98 = paddle._C_ops.transpose(reshape__426, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_378 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__428, reshape__429 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_98, full_int_array_378), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_379 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__430, reshape__431 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__428, full_int_array_379), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_61 = paddle._C_ops.shape(paddle.cast(reshape__430, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_380 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_381 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(shape_61, [0], full_int_array_380, full_int_array_381, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_116 = paddle._C_ops.matmul(reshape__430, parameter_268, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__143 = paddle._C_ops.add_(matmul_116, parameter_269)

        # pd_op.full: (1xi32) <- ()
        full_390 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_391 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_392 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_393 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_110 = [slice_118, full_390, full_391, full_392, full_393]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__432, reshape__433 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__143, [x.reshape([]) for x in combine_110]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_99 = paddle._C_ops.transpose(reshape__432, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_382 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_383 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(transpose_99, [0], full_int_array_382, full_int_array_383, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_384 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_385 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(transpose_99, [0], full_int_array_384, full_int_array_385, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_386 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_387 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(transpose_99, [0], full_int_array_386, full_int_array_387, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_394 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__19 = paddle._C_ops.scale_(slice_119, full_394, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_100 = paddle._C_ops.transpose(slice_120, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_117 = paddle._C_ops.matmul(scale__19, transpose_100, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__144 = paddle._C_ops.add_(matmul_117, parameter_270)

        # pd_op.full: (xi32) <- ()
        full_395 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_9 = paddle._C_ops.memcpy_h2d(slice_118, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__9 = paddle._C_ops.floor_divide_(memcpy_h2d_9, full_395)

        # pd_op.full: (1xi32) <- ()
        full_396 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_397 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_398 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_399 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_111 = [floor_divide__9, full_396, full_397, full_398, full_399]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__434, reshape__435 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__144, [x.reshape([]) for x in combine_111]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_388 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_18, unsqueeze_19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_271, full_int_array_388), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_389 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_18, full_int_array_389), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__145 = paddle._C_ops.add_(reshape__434, unsqueeze__18)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_390 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__436, reshape__437 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__145, full_int_array_390), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__19 = paddle._C_ops.softmax_(reshape__436, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_118 = paddle._C_ops.matmul(softmax__19, slice_121, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_101 = paddle._C_ops.transpose(matmul_118, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_400 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_401 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_112 = [slice_118, full_400, full_401]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__438, reshape__439 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_101, [x.reshape([]) for x in combine_112]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_119 = paddle._C_ops.matmul(reshape__438, parameter_272, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__146 = paddle._C_ops.add_(matmul_119, parameter_273)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_391 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__440, reshape__441 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__146, full_int_array_391), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_392 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__442, reshape__443 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__440, full_int_array_392), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_102 = paddle._C_ops.transpose(reshape__442, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_393 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__444, reshape__445 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_102, full_int_array_393), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_394 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_19 = paddle._C_ops.roll(reshape__444, full_int_array_394, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_402 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_403 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_113 = [slice_116, full_402, full_403]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__446, reshape__447 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_19, [x.reshape([]) for x in combine_113]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__147 = paddle._C_ops.add_(add__142, reshape__446)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__147, parameter_274, parameter_275, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_120 = paddle._C_ops.matmul(layer_norm_126, parameter_276, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__148 = paddle._C_ops.add_(matmul_120, parameter_277)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_19 = paddle._C_ops.gelu(add__148, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_121 = paddle._C_ops.matmul(gelu_19, parameter_278, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__149 = paddle._C_ops.add_(matmul_121, parameter_279)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__150 = paddle._C_ops.add_(add__147, add__149)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_62 = paddle._C_ops.shape(paddle.cast(add__150, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_395 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_396 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(shape_62, [0], full_int_array_395, full_int_array_396, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__150, parameter_280, parameter_281, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_404 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_405 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_406 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_114 = [slice_122, full_404, full_405, full_406]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__448, reshape__449 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_129, [x.reshape([]) for x in combine_114]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_63 = paddle._C_ops.shape(paddle.cast(reshape__448, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_397 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_398 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(shape_63, [0], full_int_array_397, full_int_array_398, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_407 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_408 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_409 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_410 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_411 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_115 = [slice_123, full_407, full_408, full_409, full_410, full_411]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__450, reshape__451 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__448, [x.reshape([]) for x in combine_115]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_103 = paddle._C_ops.transpose(reshape__450, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_399 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__452, reshape__453 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_103, full_int_array_399), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_400 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__454, reshape__455 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__452, full_int_array_400), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_64 = paddle._C_ops.shape(paddle.cast(reshape__454, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_401 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_402 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(shape_64, [0], full_int_array_401, full_int_array_402, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_122 = paddle._C_ops.matmul(reshape__454, parameter_282, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__151 = paddle._C_ops.add_(matmul_122, parameter_283)

        # pd_op.full: (1xi32) <- ()
        full_412 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_413 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_414 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_415 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_116 = [slice_124, full_412, full_413, full_414, full_415]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__456, reshape__457 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__151, [x.reshape([]) for x in combine_116]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_104 = paddle._C_ops.transpose(reshape__456, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_403 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_404 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(transpose_104, [0], full_int_array_403, full_int_array_404, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_405 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_406 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(transpose_104, [0], full_int_array_405, full_int_array_406, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_407 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_408 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(transpose_104, [0], full_int_array_407, full_int_array_408, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_416 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__20 = paddle._C_ops.scale_(slice_125, full_416, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_105 = paddle._C_ops.transpose(slice_126, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_123 = paddle._C_ops.matmul(scale__20, transpose_105, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__152 = paddle._C_ops.add_(matmul_123, parameter_284)

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__20 = paddle._C_ops.softmax_(add__152, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_124 = paddle._C_ops.matmul(softmax__20, slice_127, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_106 = paddle._C_ops.transpose(matmul_124, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_417 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_418 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_117 = [slice_124, full_417, full_418]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__458, reshape__459 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_106, [x.reshape([]) for x in combine_117]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_125 = paddle._C_ops.matmul(reshape__458, parameter_285, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__153 = paddle._C_ops.add_(matmul_125, parameter_286)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_409 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__460, reshape__461 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__153, full_int_array_409), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_410 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__462, reshape__463 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__460, full_int_array_410), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_107 = paddle._C_ops.transpose(reshape__462, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_411 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__464, reshape__465 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_107, full_int_array_411), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_419 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_420 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_118 = [slice_122, full_419, full_420]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__466, reshape__467 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__464, [x.reshape([]) for x in combine_118]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__154 = paddle._C_ops.add_(add__150, reshape__466)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__154, parameter_287, parameter_288, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_126 = paddle._C_ops.matmul(layer_norm_132, parameter_289, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__155 = paddle._C_ops.add_(matmul_126, parameter_290)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_20 = paddle._C_ops.gelu(add__155, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_127 = paddle._C_ops.matmul(gelu_20, parameter_291, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__156 = paddle._C_ops.add_(matmul_127, parameter_292)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__157 = paddle._C_ops.add_(add__154, add__156)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_65 = paddle._C_ops.shape(paddle.cast(add__157, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_412 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_413 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(shape_65, [0], full_int_array_412, full_int_array_413, [1], [0])

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__157, parameter_293, parameter_294, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_421 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_422 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_423 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_119 = [slice_128, full_421, full_422, full_423]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__468, reshape__469 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_135, [x.reshape([]) for x in combine_119]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_414 = [-6, -6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_20 = paddle._C_ops.roll(reshape__468, full_int_array_414, [1, 2])

        # pd_op.shape: (4xi32) <- (-1x24x24x512xf16)
        shape_66 = paddle._C_ops.shape(paddle.cast(roll_20, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_415 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_416 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(shape_66, [0], full_int_array_415, full_int_array_416, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_424 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_425 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_426 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_427 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_428 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_120 = [slice_129, full_424, full_425, full_426, full_427, full_428]

        # pd_op.reshape_: (-1x2x12x2x12x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__470, reshape__471 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_20, [x.reshape([]) for x in combine_120]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x2x12x12x512xf16) <- (-1x2x12x2x12x512xf16)
        transpose_108 = paddle._C_ops.transpose(reshape__470, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_417 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x2x2x12x12x512xf16) <- (-1x2x2x12x12x512xf16, 4xi64)
        reshape__472, reshape__473 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_108, full_int_array_417), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_418 = [-1, 144, 512]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 3xi64)
        reshape__474, reshape__475 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__472, full_int_array_418), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x512xf16)
        shape_67 = paddle._C_ops.shape(paddle.cast(reshape__474, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_419 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_420 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(shape_67, [0], full_int_array_419, full_int_array_420, [1], [0])

        # pd_op.matmul: (-1x144x1536xf16) <- (-1x144x512xf16, 512x1536xf16)
        matmul_128 = paddle._C_ops.matmul(reshape__474, parameter_295, False, False)

        # pd_op.add_: (-1x144x1536xf16) <- (-1x144x1536xf16, 1536xf16)
        add__158 = paddle._C_ops.add_(matmul_128, parameter_296)

        # pd_op.full: (1xi32) <- ()
        full_429 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_430 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_431 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_432 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_121 = [slice_130, full_429, full_430, full_431, full_432]

        # pd_op.reshape_: (-1x144x3x16x32xf16, 0x-1x144x1536xf16) <- (-1x144x1536xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__476, reshape__477 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__158, [x.reshape([]) for x in combine_121]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x16x144x32xf16) <- (-1x144x3x16x32xf16)
        transpose_109 = paddle._C_ops.transpose(reshape__476, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_421 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_422 = [1]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(transpose_109, [0], full_int_array_421, full_int_array_422, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_423 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_424 = [2]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(transpose_109, [0], full_int_array_423, full_int_array_424, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_425 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_426 = [3]

        # pd_op.slice: (-1x16x144x32xf16) <- (3x-1x16x144x32xf16, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(transpose_109, [0], full_int_array_425, full_int_array_426, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_433 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x16x144x32xf16) <- (-1x16x144x32xf16, 1xf32)
        scale__21 = paddle._C_ops.scale_(slice_131, full_433, float('0'), True)

        # pd_op.transpose: (-1x16x32x144xf16) <- (-1x16x144x32xf16)
        transpose_110 = paddle._C_ops.transpose(slice_132, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x144x144xf16) <- (-1x16x144x32xf16, -1x16x32x144xf16)
        matmul_129 = paddle._C_ops.matmul(scale__21, transpose_110, False, False)

        # pd_op.add_: (-1x16x144x144xf16) <- (-1x16x144x144xf16, 1x16x144x144xf16)
        add__159 = paddle._C_ops.add_(matmul_129, parameter_297)

        # pd_op.full: (xi32) <- ()
        full_434 = paddle._C_ops.full([], float('4'), paddle.int32, paddle.framework._current_expected_place())

        # pd_op.memcpy_h2d: (1xi32) <- (1xi32)
        memcpy_h2d_10 = paddle._C_ops.memcpy_h2d(slice_130, 1)

        # pd_op.floor_divide_: (1xi32) <- (1xi32, xi32)
        floor_divide__10 = paddle._C_ops.floor_divide_(memcpy_h2d_10, full_434)

        # pd_op.full: (1xi32) <- ()
        full_435 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_436 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_437 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_438 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_122 = [floor_divide__10, full_435, full_436, full_437, full_438]

        # pd_op.reshape_: (-1x4x16x144x144xf16, 0x-1x16x144x144xf16) <- (-1x16x144x144xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__478, reshape__479 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__159, [x.reshape([]) for x in combine_122]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_427 = [1]

        # pd_op.unsqueeze: (4x1x144x144xf16, None) <- (4x144x144xf16, 1xi64)
        unsqueeze_20, unsqueeze_21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_298, full_int_array_427), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_428 = [0]

        # pd_op.unsqueeze_: (1x4x1x144x144xf16, None) <- (4x1x144x144xf16, 1xi64)
        unsqueeze__20, unsqueeze__21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(unsqueeze_20, full_int_array_428), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 1x4x1x144x144xf16)
        add__160 = paddle._C_ops.add_(reshape__478, unsqueeze__20)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_429 = [-1, 16, 144, 144]

        # pd_op.reshape_: (-1x16x144x144xf16, 0x-1x4x16x144x144xf16) <- (-1x4x16x144x144xf16, 4xi64)
        reshape__480, reshape__481 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__160, full_int_array_429), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.softmax_: (-1x16x144x144xf16) <- (-1x16x144x144xf16)
        softmax__21 = paddle._C_ops.softmax_(reshape__480, -1)

        # pd_op.matmul: (-1x16x144x32xf16) <- (-1x16x144x144xf16, -1x16x144x32xf16)
        matmul_130 = paddle._C_ops.matmul(softmax__21, slice_133, False, False)

        # pd_op.transpose: (-1x144x16x32xf16) <- (-1x16x144x32xf16)
        transpose_111 = paddle._C_ops.transpose(matmul_130, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_439 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_440 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_123 = [slice_130, full_439, full_440]

        # pd_op.reshape_: (-1x144x512xf16, 0x-1x144x16x32xf16) <- (-1x144x16x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__482, reshape__483 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_111, [x.reshape([]) for x in combine_123]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x512xf16) <- (-1x144x512xf16, 512x512xf16)
        matmul_131 = paddle._C_ops.matmul(reshape__482, parameter_299, False, False)

        # pd_op.add_: (-1x144x512xf16) <- (-1x144x512xf16, 512xf16)
        add__161 = paddle._C_ops.add_(matmul_131, parameter_300)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_430 = [-1, 12, 12, 512]

        # pd_op.reshape_: (-1x12x12x512xf16, 0x-1x144x512xf16) <- (-1x144x512xf16, 4xi64)
        reshape__484, reshape__485 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__161, full_int_array_430), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_431 = [-1, 2, 2, 12, 12, 512]

        # pd_op.reshape_: (-1x2x2x12x12x512xf16, 0x-1x12x12x512xf16) <- (-1x12x12x512xf16, 6xi64)
        reshape__486, reshape__487 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__484, full_int_array_431), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x12x2x12x512xf16) <- (-1x2x2x12x12x512xf16)
        transpose_112 = paddle._C_ops.transpose(reshape__486, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_432 = [-1, 24, 24, 512]

        # pd_op.reshape_: (-1x24x24x512xf16, 0x-1x2x12x2x12x512xf16) <- (-1x2x12x2x12x512xf16, 4xi64)
        reshape__488, reshape__489 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_112, full_int_array_432), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_433 = [6, 6]

        # pd_op.roll: (-1x24x24x512xf16) <- (-1x24x24x512xf16, 2xi64)
        roll_21 = paddle._C_ops.roll(reshape__488, full_int_array_433, [1, 2])

        # pd_op.full: (1xi32) <- ()
        full_441 = paddle._C_ops.full([1], float('576'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_442 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_124 = [slice_128, full_441, full_442]

        # pd_op.reshape_: (-1x576x512xf16, 0x-1x24x24x512xf16) <- (-1x24x24x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__490, reshape__491 = (lambda x, f: f(x))(paddle._C_ops.reshape_(roll_21, [x.reshape([]) for x in combine_124]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__162 = paddle._C_ops.add_(add__157, reshape__490)

        # pd_op.layer_norm: (-1x576x512xf16, -576xf32, -576xf32) <- (-1x576x512xf16, 512xf32, 512xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__162, parameter_301, parameter_302, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x576x2048xf16) <- (-1x576x512xf16, 512x2048xf16)
        matmul_132 = paddle._C_ops.matmul(layer_norm_138, parameter_303, False, False)

        # pd_op.add_: (-1x576x2048xf16) <- (-1x576x2048xf16, 2048xf16)
        add__163 = paddle._C_ops.add_(matmul_132, parameter_304)

        # pd_op.gelu: (-1x576x2048xf16) <- (-1x576x2048xf16)
        gelu_21 = paddle._C_ops.gelu(add__163, False)

        # pd_op.matmul: (-1x576x512xf16) <- (-1x576x2048xf16, 2048x512xf16)
        matmul_133 = paddle._C_ops.matmul(gelu_21, parameter_305, False, False)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, 512xf16)
        add__164 = paddle._C_ops.add_(matmul_133, parameter_306)

        # pd_op.add_: (-1x576x512xf16) <- (-1x576x512xf16, -1x576x512xf16)
        add__165 = paddle._C_ops.add_(add__162, add__164)

        # pd_op.shape: (3xi32) <- (-1x576x512xf16)
        shape_68 = paddle._C_ops.shape(paddle.cast(add__165, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_434 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_435 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_68, [0], full_int_array_434, full_int_array_435, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_443 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_444 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_445 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_446 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_447 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_125 = [slice_134, full_443, full_444, full_445, full_446, full_447]

        # pd_op.reshape_: (-1x12x2x12x2x512xf16, 0x-1x576x512xf16) <- (-1x576x512xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__492, reshape__493 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__165, [x.reshape([]) for x in combine_125]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x12x2x2x512xf16) <- (-1x12x2x12x2x512xf16)
        transpose_113 = paddle._C_ops.transpose(reshape__492, [0, 1, 3, 4, 2, 5])

        # pd_op.full: (1xi32) <- ()
        full_448 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_449 = paddle._C_ops.full([1], float('2048'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_126 = [slice_134, full_448, full_449]

        # pd_op.reshape_: (-1x144x2048xf16, 0x-1x12x12x2x2x512xf16) <- (-1x12x12x2x2x512xf16, [1xi32, 1xi32, 1xi32])
        reshape__494, reshape__495 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_113, [x.reshape([]) for x in combine_126]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x144x2048xf16, -144xf32, -144xf32) <- (-1x144x2048xf16, 2048xf32, 2048xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__494, parameter_307, parameter_308, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x144x1024xf16) <- (-1x144x2048xf16, 2048x1024xf16)
        matmul_134 = paddle._C_ops.matmul(layer_norm_141, parameter_309, False, False)

        # pd_op.shape: (3xi32) <- (-1x144x1024xf16)
        shape_69 = paddle._C_ops.shape(paddle.cast(matmul_134, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_436 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_437 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(shape_69, [0], full_int_array_436, full_int_array_437, [1], [0])

        # pd_op.layer_norm: (-1x144x1024xf16, -144xf32, -144xf32) <- (-1x144x1024xf16, 1024xf32, 1024xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_134, parameter_310, parameter_311, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_450 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_451 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_452 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_127 = [slice_135, full_450, full_451, full_452]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x144x1024xf16) <- (-1x144x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__496, reshape__497 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_144, [x.reshape([]) for x in combine_127]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x12x12x1024xf16)
        shape_70 = paddle._C_ops.shape(paddle.cast(reshape__496, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_438 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_439 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(shape_70, [0], full_int_array_438, full_int_array_439, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_453 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_454 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_455 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_456 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_457 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_128 = [slice_136, full_453, full_454, full_455, full_456, full_457]

        # pd_op.reshape_: (-1x1x12x1x12x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__498, reshape__499 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__496, [x.reshape([]) for x in combine_128]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x1x12x12x1024xf16) <- (-1x1x12x1x12x1024xf16)
        transpose_114 = paddle._C_ops.transpose(reshape__498, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_440 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x1x1x12x12x1024xf16) <- (-1x1x1x12x12x1024xf16, 4xi64)
        reshape__500, reshape__501 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_114, full_int_array_440), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_441 = [-1, 144, 1024]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, 3xi64)
        reshape__502, reshape__503 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__500, full_int_array_441), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x1024xf16)
        shape_71 = paddle._C_ops.shape(paddle.cast(reshape__502, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_442 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_443 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(shape_71, [0], full_int_array_442, full_int_array_443, [1], [0])

        # pd_op.matmul: (-1x144x3072xf16) <- (-1x144x1024xf16, 1024x3072xf16)
        matmul_135 = paddle._C_ops.matmul(reshape__502, parameter_312, False, False)

        # pd_op.add_: (-1x144x3072xf16) <- (-1x144x3072xf16, 3072xf16)
        add__166 = paddle._C_ops.add_(matmul_135, parameter_313)

        # pd_op.full: (1xi32) <- ()
        full_458 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_459 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_460 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_461 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_129 = [slice_137, full_458, full_459, full_460, full_461]

        # pd_op.reshape_: (-1x144x3x32x32xf16, 0x-1x144x3072xf16) <- (-1x144x3072xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__504, reshape__505 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__166, [x.reshape([]) for x in combine_129]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x32x144x32xf16) <- (-1x144x3x32x32xf16)
        transpose_115 = paddle._C_ops.transpose(reshape__504, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_444 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_445 = [1]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(transpose_115, [0], full_int_array_444, full_int_array_445, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_446 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_447 = [2]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(transpose_115, [0], full_int_array_446, full_int_array_447, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_448 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_449 = [3]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(transpose_115, [0], full_int_array_448, full_int_array_449, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_462 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x32x144x32xf16) <- (-1x32x144x32xf16, 1xf32)
        scale__22 = paddle._C_ops.scale_(slice_138, full_462, float('0'), True)

        # pd_op.transpose: (-1x32x32x144xf16) <- (-1x32x144x32xf16)
        transpose_116 = paddle._C_ops.transpose(slice_139, [0, 1, 3, 2])

        # pd_op.matmul: (-1x32x144x144xf16) <- (-1x32x144x32xf16, -1x32x32x144xf16)
        matmul_136 = paddle._C_ops.matmul(scale__22, transpose_116, False, False)

        # pd_op.add_: (-1x32x144x144xf16) <- (-1x32x144x144xf16, 1x32x144x144xf16)
        add__167 = paddle._C_ops.add_(matmul_136, parameter_314)

        # pd_op.softmax_: (-1x32x144x144xf16) <- (-1x32x144x144xf16)
        softmax__22 = paddle._C_ops.softmax_(add__167, -1)

        # pd_op.matmul: (-1x32x144x32xf16) <- (-1x32x144x144xf16, -1x32x144x32xf16)
        matmul_137 = paddle._C_ops.matmul(softmax__22, slice_140, False, False)

        # pd_op.transpose: (-1x144x32x32xf16) <- (-1x32x144x32xf16)
        transpose_117 = paddle._C_ops.transpose(matmul_137, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_463 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_464 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_130 = [slice_137, full_463, full_464]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x144x32x32xf16) <- (-1x144x32x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__506, reshape__507 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_117, [x.reshape([]) for x in combine_130]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024x1024xf16)
        matmul_138 = paddle._C_ops.matmul(reshape__506, parameter_315, False, False)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024xf16)
        add__168 = paddle._C_ops.add_(matmul_138, parameter_316)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_450 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x144x1024xf16) <- (-1x144x1024xf16, 4xi64)
        reshape__508, reshape__509 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__168, full_int_array_450), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_451 = [-1, 1, 1, 12, 12, 1024]

        # pd_op.reshape_: (-1x1x1x12x12x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, 6xi64)
        reshape__510, reshape__511 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__508, full_int_array_451), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x12x1x12x1024xf16) <- (-1x1x1x12x12x1024xf16)
        transpose_118 = paddle._C_ops.transpose(reshape__510, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_452 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x1x12x1x12x1024xf16) <- (-1x1x12x1x12x1024xf16, 4xi64)
        reshape__512, reshape__513 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_118, full_int_array_452), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_465 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_466 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_131 = [slice_135, full_465, full_466]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, [1xi32, 1xi32, 1xi32])
        reshape__514, reshape__515 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__512, [x.reshape([]) for x in combine_131]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, -1x144x1024xf16)
        add__169 = paddle._C_ops.add_(matmul_134, reshape__514)

        # pd_op.layer_norm: (-1x144x1024xf16, -144xf32, -144xf32) <- (-1x144x1024xf16, 1024xf32, 1024xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__169, parameter_317, parameter_318, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x144x4096xf16) <- (-1x144x1024xf16, 1024x4096xf16)
        matmul_139 = paddle._C_ops.matmul(layer_norm_147, parameter_319, False, False)

        # pd_op.add_: (-1x144x4096xf16) <- (-1x144x4096xf16, 4096xf16)
        add__170 = paddle._C_ops.add_(matmul_139, parameter_320)

        # pd_op.gelu: (-1x144x4096xf16) <- (-1x144x4096xf16)
        gelu_22 = paddle._C_ops.gelu(add__170, False)

        # pd_op.matmul: (-1x144x1024xf16) <- (-1x144x4096xf16, 4096x1024xf16)
        matmul_140 = paddle._C_ops.matmul(gelu_22, parameter_321, False, False)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024xf16)
        add__171 = paddle._C_ops.add_(matmul_140, parameter_322)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, -1x144x1024xf16)
        add__172 = paddle._C_ops.add_(add__169, add__171)

        # pd_op.shape: (3xi32) <- (-1x144x1024xf16)
        shape_72 = paddle._C_ops.shape(paddle.cast(add__172, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_453 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_454 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(shape_72, [0], full_int_array_453, full_int_array_454, [1], [0])

        # pd_op.layer_norm: (-1x144x1024xf16, -144xf32, -144xf32) <- (-1x144x1024xf16, 1024xf32, 1024xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__172, parameter_323, parameter_324, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_467 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_468 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_469 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_132 = [slice_141, full_467, full_468, full_469]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x144x1024xf16) <- (-1x144x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__516, reshape__517 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_150, [x.reshape([]) for x in combine_132]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (4xi32) <- (-1x12x12x1024xf16)
        shape_73 = paddle._C_ops.shape(paddle.cast(reshape__516, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_455 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_456 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(shape_73, [0], full_int_array_455, full_int_array_456, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_470 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_471 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_472 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_473 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_474 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_133 = [slice_142, full_470, full_471, full_472, full_473, full_474]

        # pd_op.reshape_: (-1x1x12x1x12x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__518, reshape__519 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__516, [x.reshape([]) for x in combine_133]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x1x12x12x1024xf16) <- (-1x1x12x1x12x1024xf16)
        transpose_119 = paddle._C_ops.transpose(reshape__518, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_457 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x1x1x12x12x1024xf16) <- (-1x1x1x12x12x1024xf16, 4xi64)
        reshape__520, reshape__521 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_119, full_int_array_457), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_458 = [-1, 144, 1024]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, 3xi64)
        reshape__522, reshape__523 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__520, full_int_array_458), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.shape: (3xi32) <- (-1x144x1024xf16)
        shape_74 = paddle._C_ops.shape(paddle.cast(reshape__522, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_459 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_460 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(shape_74, [0], full_int_array_459, full_int_array_460, [1], [0])

        # pd_op.matmul: (-1x144x3072xf16) <- (-1x144x1024xf16, 1024x3072xf16)
        matmul_141 = paddle._C_ops.matmul(reshape__522, parameter_325, False, False)

        # pd_op.add_: (-1x144x3072xf16) <- (-1x144x3072xf16, 3072xf16)
        add__173 = paddle._C_ops.add_(matmul_141, parameter_326)

        # pd_op.full: (1xi32) <- ()
        full_475 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_476 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_477 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_478 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_134 = [slice_143, full_475, full_476, full_477, full_478]

        # pd_op.reshape_: (-1x144x3x32x32xf16, 0x-1x144x3072xf16) <- (-1x144x3072xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__524, reshape__525 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__173, [x.reshape([]) for x in combine_134]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (3x-1x32x144x32xf16) <- (-1x144x3x32x32xf16)
        transpose_120 = paddle._C_ops.transpose(reshape__524, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_461 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_462 = [1]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(transpose_120, [0], full_int_array_461, full_int_array_462, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_463 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_464 = [2]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(transpose_120, [0], full_int_array_463, full_int_array_464, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_465 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_466 = [3]

        # pd_op.slice: (-1x32x144x32xf16) <- (3x-1x32x144x32xf16, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(transpose_120, [0], full_int_array_465, full_int_array_466, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_479 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x32x144x32xf16) <- (-1x32x144x32xf16, 1xf32)
        scale__23 = paddle._C_ops.scale_(slice_144, full_479, float('0'), True)

        # pd_op.transpose: (-1x32x32x144xf16) <- (-1x32x144x32xf16)
        transpose_121 = paddle._C_ops.transpose(slice_145, [0, 1, 3, 2])

        # pd_op.matmul: (-1x32x144x144xf16) <- (-1x32x144x32xf16, -1x32x32x144xf16)
        matmul_142 = paddle._C_ops.matmul(scale__23, transpose_121, False, False)

        # pd_op.add_: (-1x32x144x144xf16) <- (-1x32x144x144xf16, 1x32x144x144xf16)
        add__174 = paddle._C_ops.add_(matmul_142, parameter_327)

        # pd_op.softmax_: (-1x32x144x144xf16) <- (-1x32x144x144xf16)
        softmax__23 = paddle._C_ops.softmax_(add__174, -1)

        # pd_op.matmul: (-1x32x144x32xf16) <- (-1x32x144x144xf16, -1x32x144x32xf16)
        matmul_143 = paddle._C_ops.matmul(softmax__23, slice_146, False, False)

        # pd_op.transpose: (-1x144x32x32xf16) <- (-1x32x144x32xf16)
        transpose_122 = paddle._C_ops.transpose(matmul_143, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_480 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_481 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_135 = [slice_143, full_480, full_481]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x144x32x32xf16) <- (-1x144x32x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__526, reshape__527 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_122, [x.reshape([]) for x in combine_135]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024x1024xf16)
        matmul_144 = paddle._C_ops.matmul(reshape__526, parameter_328, False, False)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024xf16)
        add__175 = paddle._C_ops.add_(matmul_144, parameter_329)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_467 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x144x1024xf16) <- (-1x144x1024xf16, 4xi64)
        reshape__528, reshape__529 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__175, full_int_array_467), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_468 = [-1, 1, 1, 12, 12, 1024]

        # pd_op.reshape_: (-1x1x1x12x12x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, 6xi64)
        reshape__530, reshape__531 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__528, full_int_array_468), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x1x12x1x12x1024xf16) <- (-1x1x1x12x12x1024xf16)
        transpose_123 = paddle._C_ops.transpose(reshape__530, [0, 1, 3, 2, 4, 5])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_469 = [-1, 12, 12, 1024]

        # pd_op.reshape_: (-1x12x12x1024xf16, 0x-1x1x12x1x12x1024xf16) <- (-1x1x12x1x12x1024xf16, 4xi64)
        reshape__532, reshape__533 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_123, full_int_array_469), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_482 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_483 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_136 = [slice_141, full_482, full_483]

        # pd_op.reshape_: (-1x144x1024xf16, 0x-1x12x12x1024xf16) <- (-1x12x12x1024xf16, [1xi32, 1xi32, 1xi32])
        reshape__534, reshape__535 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__532, [x.reshape([]) for x in combine_136]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, -1x144x1024xf16)
        add__176 = paddle._C_ops.add_(add__172, reshape__534)

        # pd_op.layer_norm: (-1x144x1024xf16, -144xf32, -144xf32) <- (-1x144x1024xf16, 1024xf32, 1024xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__176, parameter_330, parameter_331, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x144x4096xf16) <- (-1x144x1024xf16, 1024x4096xf16)
        matmul_145 = paddle._C_ops.matmul(layer_norm_153, parameter_332, False, False)

        # pd_op.add_: (-1x144x4096xf16) <- (-1x144x4096xf16, 4096xf16)
        add__177 = paddle._C_ops.add_(matmul_145, parameter_333)

        # pd_op.gelu: (-1x144x4096xf16) <- (-1x144x4096xf16)
        gelu_23 = paddle._C_ops.gelu(add__177, False)

        # pd_op.matmul: (-1x144x1024xf16) <- (-1x144x4096xf16, 4096x1024xf16)
        matmul_146 = paddle._C_ops.matmul(gelu_23, parameter_334, False, False)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, 1024xf16)
        add__178 = paddle._C_ops.add_(matmul_146, parameter_335)

        # pd_op.add_: (-1x144x1024xf16) <- (-1x144x1024xf16, -1x144x1024xf16)
        add__179 = paddle._C_ops.add_(add__176, add__178)

        # pd_op.layer_norm: (-1x144x1024xf16, -144xf32, -144xf32) <- (-1x144x1024xf16, 1024xf32, 1024xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__179, parameter_336, parameter_337, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.transpose: (-1x1024x144xf16) <- (-1x144x1024xf16)
        transpose_124 = paddle._C_ops.transpose(layer_norm_156, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_470 = [2]

        # pd_op.unsqueeze_: (-1x1024x1x144xf16, None) <- (-1x1024x144xf16, 1xi64)
        unsqueeze__22, unsqueeze__23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_124, full_int_array_470), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_471 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf16) <- (-1x1024x1x144xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(unsqueeze__22, full_int_array_471, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_472 = [2]

        # pd_op.squeeze_: (-1x1024x1xf16, None) <- (-1x1024x1x1xf16, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_0, full_int_array_472), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.flatten_: (-1x1024xf16, None) <- (-1x1024x1xf16)
        flatten__2, flatten__3 = (lambda x, f: f(x))(paddle._C_ops.flatten_(squeeze__0, 1, 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x1024xf16, 1024x1000xf16)
        matmul_147 = paddle._C_ops.matmul(flatten__2, parameter_338, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__180 = paddle._C_ops.add_(matmul_147, parameter_339)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__24 = paddle._C_ops.softmax_(add__180, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__24, paddle.float32)
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

    def forward(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_26, parameter_25, parameter_27, parameter_28, parameter_29, parameter_30, parameter_32, parameter_31, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_56, parameter_55, parameter_57, parameter_58, parameter_59, parameter_60, parameter_62, parameter_61, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_86, parameter_85, parameter_87, parameter_88, parameter_89, parameter_90, parameter_92, parameter_91, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_126, parameter_125, parameter_127, parameter_128, parameter_129, parameter_130, parameter_132, parameter_131, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_142, parameter_143, parameter_144, parameter_146, parameter_145, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_153, parameter_152, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_194, parameter_193, parameter_195, parameter_196, parameter_197, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_234, parameter_233, parameter_235, parameter_236, parameter_237, parameter_238, parameter_240, parameter_239, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_248, parameter_247, parameter_249, parameter_250, parameter_251, parameter_252, parameter_254, parameter_253, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_281, parameter_280, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_288, parameter_287, parameter_289, parameter_290, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_311, parameter_310, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_318, parameter_317, parameter_319, parameter_320, parameter_321, parameter_322, parameter_324, parameter_323, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, feed_0):
        return self.builtin_module_2575_0_0(parameter_0, parameter_1, parameter_3, parameter_2, parameter_5, parameter_4, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_26, parameter_25, parameter_27, parameter_28, parameter_29, parameter_30, parameter_32, parameter_31, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_42, parameter_41, parameter_43, parameter_44, parameter_45, parameter_46, parameter_48, parameter_47, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_56, parameter_55, parameter_57, parameter_58, parameter_59, parameter_60, parameter_62, parameter_61, parameter_63, parameter_65, parameter_64, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_86, parameter_85, parameter_87, parameter_88, parameter_89, parameter_90, parameter_92, parameter_91, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_113, parameter_112, parameter_114, parameter_115, parameter_116, parameter_117, parameter_119, parameter_118, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_126, parameter_125, parameter_127, parameter_128, parameter_129, parameter_130, parameter_132, parameter_131, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_142, parameter_143, parameter_144, parameter_146, parameter_145, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_153, parameter_152, parameter_154, parameter_155, parameter_156, parameter_157, parameter_159, parameter_158, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_167, parameter_166, parameter_168, parameter_169, parameter_170, parameter_171, parameter_173, parameter_172, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_194, parameter_193, parameter_195, parameter_196, parameter_197, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_207, parameter_206, parameter_208, parameter_209, parameter_210, parameter_211, parameter_213, parameter_212, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_234, parameter_233, parameter_235, parameter_236, parameter_237, parameter_238, parameter_240, parameter_239, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_248, parameter_247, parameter_249, parameter_250, parameter_251, parameter_252, parameter_254, parameter_253, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_272, parameter_273, parameter_275, parameter_274, parameter_276, parameter_277, parameter_278, parameter_279, parameter_281, parameter_280, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_288, parameter_287, parameter_289, parameter_290, parameter_291, parameter_292, parameter_294, parameter_293, parameter_295, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_311, parameter_310, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_318, parameter_317, parameter_319, parameter_320, parameter_321, parameter_322, parameter_324, parameter_323, parameter_325, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2575_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([128, 3, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128, 384], dtype='float16', min=0, max=0.5),
            # parameter_7
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([1, 4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128, 512], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512, 128], dtype='float16', min=0, max=0.5),
            # parameter_16
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128, 384], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([1, 4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128, 128], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([128, 512], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([512, 128], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([512, 256], dtype='float16', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256, 768], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1, 8, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256, 1024], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_45
            paddle.uniform([1024, 256], dtype='float16', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256, 768], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([1, 8, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256, 256], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256, 1024], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([1024, 256], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([1024, 512], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_176
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_250
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([512, 1536], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([1536], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([1, 16, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([4, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512, 512], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512, 2048], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([2048, 512], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([2048, 1024], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1024, 3072], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1, 32, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024, 1024], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1024, 4096], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([4096], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([4096, 1024], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1024, 3072], dtype='float16', min=0, max=0.5),
            # parameter_326
            paddle.uniform([3072], dtype='float16', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1, 32, 144, 144], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1024, 1024], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_331
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1024, 4096], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([4096], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([4096, 1024], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1024, 1000], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 384, 384], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[128, 3, 4, 4], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128, 384], dtype='float16'),
            # parameter_7
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[1, 4, 144, 144], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128, 512], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[512, 128], dtype='float16'),
            # parameter_16
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[128, 384], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[1, 4, 144, 144], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 144, 144], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[128, 128], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[128, 512], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[512, 128], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[512, 256], dtype='float16'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256, 768], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[1, 8, 144, 144], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256, 1024], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_45
            paddle.static.InputSpec(shape=[1024, 256], dtype='float16'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[256, 768], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[1, 8, 144, 144], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[16, 144, 144], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[256, 256], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256, 1024], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[1024, 256], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[1024, 512], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_175
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_176
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_250
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_252
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_265
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_267
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_272
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_281
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[512, 1536], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[1536], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[1, 16, 144, 144], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[4, 144, 144], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[512, 512], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[512, 2048], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[2048, 512], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[2048, 1024], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[1024, 3072], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[1, 32, 144, 144], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[1024, 4096], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[4096], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[4096, 1024], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[1024, 3072], dtype='float16'),
            # parameter_326
            paddle.static.InputSpec(shape=[3072], dtype='float16'),
            # parameter_327
            paddle.static.InputSpec(shape=[1, 32, 144, 144], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_331
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[1024, 4096], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[4096], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[4096, 1024], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[1024, 1000], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 384, 384], dtype='float32'),
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