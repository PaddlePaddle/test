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
    return [1071][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1414_0_0(self, parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_305, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_335, feed_0):

        # pd_op.transpose: (1x1x25x2x350xf32) <- (1x2x350x25x1xf32)
        transpose_0 = paddle._C_ops.transpose(feed_0, [0, 4, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [1, 50, 350]

        # pd_op.reshape_: (1x50x350xf32, 0x1x1x25x2x350xf32) <- (1x1x25x2x350xf32, 3xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (1x50x350xf32, 50xf32, 50xf32, xf32, xf32, None) <- (1x50x350xf32, 50xf32, 50xf32, 50xf32, 50xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__0, parameter_0, parameter_1, parameter_2, parameter_3, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_1 = [1, 1, 25, 2, 350]

        # pd_op.reshape_: (1x1x25x2x350xf32, 0x1x50x350xf32) <- (1x50x350xf32, 5xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x1x2x350x25xf32) <- (1x1x25x2x350xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 1, 3, 4, 2])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 2, 350, 25]

        # pd_op.reshape_: (1x2x350x25xf32, 0x1x1x2x350x25xf32) <- (1x1x2x350x25xf32, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_0 = paddle._C_ops.add(parameter_4, parameter_5)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape__4, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_2 = paddle._C_ops.transpose(add__0, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(reshape__4, parameter_8, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_0 = paddle._C_ops.matmul(reshape__6, reshape__8, False, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_0, full_0, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(add_0, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__2 = paddle._C_ops.add_(softmax__0, slice_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [1, 700, 25]

        # pd_op.reshape: (1x700x25xf32, 0x1x2x350x25xf32) <- (1x2x350x25xf32, 3xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf32) <- (1x700x25xf32, 1x25x25xf32)
        matmul_1 = paddle._C_ops.matmul(reshape_4, add__2, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 2, 350, 25]

        # pd_op.reshape_: (1x2x350x25xf32, 0x1x700x25xf32) <- (1x700x25xf32, 4xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_1, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x2x350x25xf32, 64x2x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(reshape__10, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_2, reshape_6)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(reshape__4, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_3, reshape_8)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_3 = paddle._C_ops.transpose(add__4, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_13 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(reshape__4, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_4, reshape_10)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_15 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__5, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_2 = paddle._C_ops.matmul(reshape__12, reshape__14, False, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_2, full_1, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(add_0, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__6 = paddle._C_ops.add_(softmax__1, slice_1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_18 = [1, 700, 25]

        # pd_op.reshape: (1x700x25xf32, 0x1x2x350x25xf32) <- (1x2x350x25xf32, 3xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf32) <- (1x700x25xf32, 1x25x25xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_12, add__6, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 2, 350, 25]

        # pd_op.reshape_: (1x2x350x25xf32, 0x1x700x25xf32) <- (1x700x25xf32, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_3, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x2x350x25xf32, 64x2x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(reshape__16, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_5, reshape_14)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__8 = paddle._C_ops.add_(add__7, add__3)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(reshape__4, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_6, reshape_16)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_4 = paddle._C_ops.transpose(add__9, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_22 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x2x350x25xf32, 16x2x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(reshape__4, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_7, reshape_18)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_24 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_4 = paddle._C_ops.matmul(reshape__18, reshape__20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_4, full_2, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(add_0, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__11 = paddle._C_ops.add_(softmax__2, slice_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_27 = [1, 700, 25]

        # pd_op.reshape: (1x700x25xf32, 0x1x2x350x25xf32) <- (1x2x350x25xf32, 3xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(reshape__4, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x700x25xf32) <- (1x700x25xf32, 1x25x25xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_20, add__11, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 2, 350, 25]

        # pd_op.reshape_: (1x2x350x25xf32, 0x1x700x25xf32) <- (1x700x25xf32, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_5, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x2x350x25xf32, 64x2x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(reshape__22, parameter_22, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_23, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_8, reshape_22)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__13 = paddle._C_ops.add_(add__12, add__8)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__13, parameter_24, parameter_25, parameter_26, parameter_27, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x2x350x25xf32, 64x2x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(reshape__4, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_9, reshape_24)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__14, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__15 = paddle._C_ops.add_(batch_norm__6, batch_norm__12)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__0 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__0, parameter_34, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_35, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_10, reshape_26)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(batch_norm__18, full_3, float('0'), True)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__1 = paddle._C_ops.relu_(scale__3)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_1 = paddle._C_ops.add(parameter_40, parameter_41)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__1, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_43, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_11, reshape_28)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_5 = paddle._C_ops.transpose(add__17, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_33 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__1, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_45, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_12, reshape_30)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_35 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_6 = paddle._C_ops.matmul(reshape__24, reshape__26, False, False)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_6, full_4, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__3 = paddle._C_ops.softmax_(scale__4, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(add_1, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__19 = paddle._C_ops.add_(softmax__3, slice_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_38 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_32, add__19, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_7, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(reshape__28, parameter_46, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_47, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_13, reshape_34)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__1, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_49, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_14, reshape_36)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_6 = paddle._C_ops.transpose(add__21, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_42 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__1, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_51, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_15, reshape_38)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_44 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_8 = paddle._C_ops.matmul(reshape__30, reshape__32, False, False)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_8, full_5, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__4 = paddle._C_ops.softmax_(scale__5, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(add_1, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__23 = paddle._C_ops.add_(softmax__4, slice_4)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_47 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_40, add__23, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_9, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(reshape__34, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_53, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_16, reshape_42)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__25 = paddle._C_ops.add_(add__24, add__20)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__1, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_17, reshape_44)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_7 = paddle._C_ops.transpose(add__26, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_51 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu__1, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_18, reshape_46)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_53 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__27, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_10 = paddle._C_ops.matmul(reshape__36, reshape__38, False, False)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_10, full_6, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__5 = paddle._C_ops.softmax_(scale__6, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(add_1, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__28 = paddle._C_ops.add_(softmax__5, slice_5)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_56 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__1, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_11 = paddle._C_ops.matmul(reshape_48, add__28, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_11, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(reshape__40, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_59, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_19, reshape_50)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__30 = paddle._C_ops.add_(add__29, add__25)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__30, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__31 = paddle._C_ops.add_(batch_norm__24, relu__1)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__2 = paddle._C_ops.relu_(add__31)

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__2, parameter_64, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_20, reshape_52)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__32, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__33 = paddle._C_ops.add_(batch_norm__30, relu__1)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__3 = paddle._C_ops.relu_(add__33)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_2 = paddle._C_ops.add(parameter_70, parameter_71)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__3, parameter_72, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_73, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_21, reshape_54)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_8 = paddle._C_ops.transpose(add__34, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_61 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__3, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_75, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_22, reshape_56)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_63 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__35, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_12 = paddle._C_ops.matmul(reshape__42, reshape__44, False, False)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_12, full_7, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__6 = paddle._C_ops.softmax_(scale__7, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(add_2, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__36 = paddle._C_ops.add_(softmax__6, slice_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_66 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_58, add__36, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_13, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(reshape__46, parameter_76, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_77, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_23, reshape_60)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__3, parameter_78, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_79, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_24, reshape_62)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_9 = paddle._C_ops.transpose(add__38, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_70 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__3, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_71 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_81, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_25, reshape_64)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_72 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_14 = paddle._C_ops.matmul(reshape__48, reshape__50, False, False)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_14, full_8, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__7 = paddle._C_ops.softmax_(scale__8, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(add_2, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__40 = paddle._C_ops.add_(softmax__7, slice_7)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_75 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_66, add__40, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_76 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_15, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(reshape__52, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_77 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_83, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__41 = paddle._C_ops.add_(conv2d_26, reshape_68)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__42 = paddle._C_ops.add_(add__41, add__37)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__3, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_78 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_85, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_27, reshape_70)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_10 = paddle._C_ops.transpose(add__43, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_79 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__3, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_87, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_28, reshape_72)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_81 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_16 = paddle._C_ops.matmul(reshape__54, reshape__56, False, False)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_16, full_9, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__8 = paddle._C_ops.softmax_(scale__9, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(add_2, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__45 = paddle._C_ops.add_(softmax__8, slice_8)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_84 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__3, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_17 = paddle._C_ops.matmul(reshape_74, add__45, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_17, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(reshape__58, parameter_88, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_29, reshape_76)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__47 = paddle._C_ops.add_(add__46, add__42)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__47, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__48 = paddle._C_ops.add_(batch_norm__36, relu__3)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__4 = paddle._C_ops.relu_(add__48)

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__4, parameter_94, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_95, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__49 = paddle._C_ops.add_(conv2d_30, reshape_78)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__49, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__50 = paddle._C_ops.add_(batch_norm__42, relu__3)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__5 = paddle._C_ops.relu_(add__50)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_3 = paddle._C_ops.add(parameter_100, parameter_101)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__5, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_88 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__51 = paddle._C_ops.add_(conv2d_31, reshape_80)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_11 = paddle._C_ops.transpose(add__51, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_89 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__5, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_90 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__52 = paddle._C_ops.add_(conv2d_32, reshape_82)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_91 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__52, full_int_array_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_18 = paddle._C_ops.matmul(reshape__60, reshape__62, False, False)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_18, full_10, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__9 = paddle._C_ops.softmax_(scale__10, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(add_3, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__53 = paddle._C_ops.add_(softmax__9, slice_9)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_94 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_84, add__53, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_95 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_19, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(reshape__64, parameter_106, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_96 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_107, full_int_array_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__54 = paddle._C_ops.add_(conv2d_33, reshape_86)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__5, parameter_108, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_97 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_109, full_int_array_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__55 = paddle._C_ops.add_(conv2d_34, reshape_88)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_12 = paddle._C_ops.transpose(add__55, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_98 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__5, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_111, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__56 = paddle._C_ops.add_(conv2d_35, reshape_90)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_100 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_20 = paddle._C_ops.matmul(reshape__66, reshape__68, False, False)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_20, full_11, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__10 = paddle._C_ops.softmax_(scale__11, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(add_3, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__57 = paddle._C_ops.add_(softmax__10, slice_10)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_103 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_92, add__57, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_104 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_21, full_int_array_104), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(reshape__70, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_105 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__58 = paddle._C_ops.add_(conv2d_36, reshape_94)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__59 = paddle._C_ops.add_(add__58, add__54)

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu__5, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_106 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__60 = paddle._C_ops.add_(conv2d_37, reshape_96)

        # pd_op.transpose: (1x25x16x350xf32) <- (1x16x350x25xf32)
        transpose_13 = paddle._C_ops.transpose(add__60, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_107 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x16x350xf32) <- (1x25x16x350xf32, 3xi64)
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x350x25xf32) <- (1x64x350x25xf32, 16x64x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__5, parameter_116, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_108 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_117, full_int_array_108), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x350x25xf32) <- (1x16x350x25xf32, 1x16x1x1xf32)
        add__61 = paddle._C_ops.add_(conv2d_38, reshape_98)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_109 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x16x350x25xf32) <- (1x16x350x25xf32, 3xi64)
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__61, full_int_array_109), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_22 = paddle._C_ops.matmul(reshape__72, reshape__74, False, False)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_22, full_12, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__11 = paddle._C_ops.softmax_(scale__12, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(add_3, [0], full_int_array_110, full_int_array_111, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__62 = paddle._C_ops.add_(softmax__11, slice_11)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_112 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__5, full_int_array_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_23 = paddle._C_ops.matmul(reshape_100, add__62, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_113 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_23, full_int_array_113), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(reshape__76, parameter_118, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_114 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_119, full_int_array_114), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__63 = paddle._C_ops.add_(conv2d_39, reshape_102)

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__64 = paddle._C_ops.add_(add__63, add__59)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__64, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__65 = paddle._C_ops.add_(batch_norm__48, relu__5)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__6 = paddle._C_ops.relu_(add__65)

        # pd_op.conv2d: (1x64x350x25xf32) <- (1x64x350x25xf32, 64x64x9x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__6, parameter_124, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_115 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_125, full_int_array_115), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x1x1xf32)
        add__66 = paddle._C_ops.add_(conv2d_40, reshape_104)

        # pd_op.batch_norm_: (1x64x350x25xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x350x25xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__66, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x350x25xf32) <- (1x64x350x25xf32, 1x64x350x25xf32)
        add__67 = paddle._C_ops.add_(batch_norm__54, relu__5)

        # pd_op.relu_: (1x64x350x25xf32) <- (1x64x350x25xf32)
        relu__7 = paddle._C_ops.relu_(add__67)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_4 = paddle._C_ops.add(parameter_130, parameter_131)

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu__7, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_116 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_133, full_int_array_116), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__68 = paddle._C_ops.add_(conv2d_41, reshape_106)

        # pd_op.transpose: (1x25x32x350xf32) <- (1x32x350x25xf32)
        transpose_14 = paddle._C_ops.transpose(add__68, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_117 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x32x350xf32) <- (1x25x32x350xf32, 3xi64)
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, full_int_array_117), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu__7, parameter_134, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_118 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_135, full_int_array_118), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__69 = paddle._C_ops.add_(conv2d_42, reshape_108)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_119 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x32x350x25xf32) <- (1x32x350x25xf32, 3xi64)
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__69, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_24 = paddle._C_ops.matmul(reshape__78, reshape__80, False, False)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_24, full_13, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__12 = paddle._C_ops.softmax_(scale__13, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(add_4, [0], full_int_array_120, full_int_array_121, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__70 = paddle._C_ops.add_(softmax__12, slice_12)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_122 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, full_int_array_122), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_110, add__70, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_123 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_25, full_int_array_123), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf32) <- (1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(reshape__82, parameter_136, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_124 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_137, full_int_array_124), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x1x1xf32)
        add__71 = paddle._C_ops.add_(conv2d_43, reshape_112)

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__7, parameter_138, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_125 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_125), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__72 = paddle._C_ops.add_(conv2d_44, reshape_114)

        # pd_op.transpose: (1x25x32x350xf32) <- (1x32x350x25xf32)
        transpose_15 = paddle._C_ops.transpose(add__72, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_126 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x32x350xf32) <- (1x25x32x350xf32, 3xi64)
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, full_int_array_126), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu__7, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_127 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_141, full_int_array_127), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__73 = paddle._C_ops.add_(conv2d_45, reshape_116)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_128 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x32x350x25xf32) <- (1x32x350x25xf32, 3xi64)
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__73, full_int_array_128), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_26 = paddle._C_ops.matmul(reshape__84, reshape__86, False, False)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_26, full_14, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__13 = paddle._C_ops.softmax_(scale__14, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add_4, [0], full_int_array_129, full_int_array_130, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__74 = paddle._C_ops.add_(softmax__13, slice_13)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_131 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, full_int_array_131), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_118, add__74, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_132 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_27, full_int_array_132), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf32) <- (1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(reshape__88, parameter_142, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_133 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_143, full_int_array_133), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x1x1xf32)
        add__75 = paddle._C_ops.add_(conv2d_46, reshape_120)

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x350x25xf32)
        add__76 = paddle._C_ops.add_(add__75, add__71)

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__7, parameter_144, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_134 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_145, full_int_array_134), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__77 = paddle._C_ops.add_(conv2d_47, reshape_122)

        # pd_op.transpose: (1x25x32x350xf32) <- (1x32x350x25xf32)
        transpose_16 = paddle._C_ops.transpose(add__77, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_135 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x32x350xf32) <- (1x25x32x350xf32, 3xi64)
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, full_int_array_135), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x350x25xf32) <- (1x64x350x25xf32, 32x64x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__7, parameter_146, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_136 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_147, full_int_array_136), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x350x25xf32) <- (1x32x350x25xf32, 1x32x1x1xf32)
        add__78 = paddle._C_ops.add_(conv2d_48, reshape_124)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_137 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x32x350x25xf32) <- (1x32x350x25xf32, 3xi64)
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__78, full_int_array_137), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_28 = paddle._C_ops.matmul(reshape__90, reshape__92, False, False)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_28, full_15, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__14 = paddle._C_ops.softmax_(scale__15, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(add_4, [0], full_int_array_138, full_int_array_139, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__79 = paddle._C_ops.add_(softmax__14, slice_14)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_140 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x64x350x25xf32) <- (1x64x350x25xf32, 3xi64)
        reshape_126, reshape_127 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__7, full_int_array_140), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_126, add__79, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_141 = [1, 64, 350, 25]

        # pd_op.reshape_: (1x64x350x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_29, full_int_array_141), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x350x25xf32) <- (1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(reshape__94, parameter_148, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_142 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_128, reshape_129 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_149, full_int_array_142), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x1x1xf32)
        add__80 = paddle._C_ops.add_(conv2d_49, reshape_128)

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x350x25xf32)
        add__81 = paddle._C_ops.add_(add__80, add__76)

        # pd_op.batch_norm_: (1x128x350x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x350x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__81, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x350x25xf32) <- (1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu__7, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_143 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_130, reshape_131 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_155, full_int_array_143), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x1x1xf32)
        add__82 = paddle._C_ops.add_(conv2d_50, reshape_130)

        # pd_op.batch_norm_: (1x128x350x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x350x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__82, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x350x25xf32) <- (1x128x350x25xf32, 1x128x350x25xf32)
        add__83 = paddle._C_ops.add_(batch_norm__60, batch_norm__66)

        # pd_op.relu_: (1x128x350x25xf32) <- (1x128x350x25xf32)
        relu__8 = paddle._C_ops.relu_(add__83)

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x350x25xf32, 128x128x9x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__8, parameter_160, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_144 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_132, reshape_133 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_144), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__84 = paddle._C_ops.add_(conv2d_51, reshape_132)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__84, parameter_162, parameter_163, parameter_164, parameter_165, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x64x350x25xf32, 128x64x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__7, parameter_166, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_145 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_134, reshape_135 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_145), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__85 = paddle._C_ops.add_(conv2d_52, reshape_134)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__85, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__86 = paddle._C_ops.add_(batch_norm__72, batch_norm__78)

        # pd_op.relu_: (1x128x175x25xf32) <- (1x128x175x25xf32)
        relu__9 = paddle._C_ops.relu_(add__86)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_5 = paddle._C_ops.add(parameter_172, parameter_173)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__9, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_146 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_136, reshape_137 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_175, full_int_array_146), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__87 = paddle._C_ops.add_(conv2d_53, reshape_136)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_17 = paddle._C_ops.transpose(add__87, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_147 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, full_int_array_147), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__9, parameter_176, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_148 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_138, reshape_139 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_177, full_int_array_148), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__88 = paddle._C_ops.add_(conv2d_54, reshape_138)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_149 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__88, full_int_array_149), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_30 = paddle._C_ops.matmul(reshape__96, reshape__98, False, False)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_30, full_16, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__15 = paddle._C_ops.softmax_(scale__16, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(add_5, [0], full_int_array_150, full_int_array_151, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__89 = paddle._C_ops.add_(softmax__15, slice_15)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_152 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_140, reshape_141 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, full_int_array_152), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_31 = paddle._C_ops.matmul(reshape_140, add__89, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_153 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_31, full_int_array_153), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(reshape__100, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_154 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_142, reshape_143 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_179, full_int_array_154), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__90 = paddle._C_ops.add_(conv2d_55, reshape_142)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu__9, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_155 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_144, reshape_145 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_155), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__91 = paddle._C_ops.add_(conv2d_56, reshape_144)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_18 = paddle._C_ops.transpose(add__91, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_156 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_18, full_int_array_156), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu__9, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_157 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_146, reshape_147 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_157), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__92 = paddle._C_ops.add_(conv2d_57, reshape_146)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_158 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__92, full_int_array_158), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_32 = paddle._C_ops.matmul(reshape__102, reshape__104, False, False)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_32, full_17, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__16 = paddle._C_ops.softmax_(scale__17, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(add_5, [0], full_int_array_159, full_int_array_160, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__93 = paddle._C_ops.add_(softmax__16, slice_16)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_161 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_148, reshape_149 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, full_int_array_161), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_148, add__93, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_162 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_33, full_int_array_162), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(reshape__106, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_163 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_150, reshape_151 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_163), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__94 = paddle._C_ops.add_(conv2d_58, reshape_150)

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__95 = paddle._C_ops.add_(add__94, add__90)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(relu__9, parameter_186, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_164 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_152, reshape_153 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_187, full_int_array_164), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__96 = paddle._C_ops.add_(conv2d_59, reshape_152)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_19 = paddle._C_ops.transpose(add__96, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_165 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, full_int_array_165), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(relu__9, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_166 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_154, reshape_155 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_166), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__97 = paddle._C_ops.add_(conv2d_60, reshape_154)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_167 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__97, full_int_array_167), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_34 = paddle._C_ops.matmul(reshape__108, reshape__110, False, False)

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_34, full_18, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__17 = paddle._C_ops.softmax_(scale__18, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(add_5, [0], full_int_array_168, full_int_array_169, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__98 = paddle._C_ops.add_(softmax__17, slice_17)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_170 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_156, reshape_157 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__9, full_int_array_170), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_35 = paddle._C_ops.matmul(reshape_156, add__98, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_171 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_35, full_int_array_171), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(reshape__112, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_172 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_158, reshape_159 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_172), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__99 = paddle._C_ops.add_(conv2d_61, reshape_158)

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__100 = paddle._C_ops.add_(add__99, add__95)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__100, parameter_192, parameter_193, parameter_194, parameter_195, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__101 = paddle._C_ops.add_(batch_norm__84, relu__9)

        # pd_op.relu_: (1x128x175x25xf32) <- (1x128x175x25xf32)
        relu__10 = paddle._C_ops.relu_(add__101)

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x9x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu__10, parameter_196, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_173 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_160, reshape_161 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_197, full_int_array_173), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__102 = paddle._C_ops.add_(conv2d_62, reshape_160)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__102, parameter_198, parameter_199, parameter_200, parameter_201, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__103 = paddle._C_ops.add_(batch_norm__90, relu__9)

        # pd_op.relu_: (1x128x175x25xf32) <- (1x128x175x25xf32)
        relu__11 = paddle._C_ops.relu_(add__103)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_6 = paddle._C_ops.add(parameter_202, parameter_203)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(relu__11, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_174 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_162, reshape_163 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_205, full_int_array_174), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__104 = paddle._C_ops.add_(conv2d_63, reshape_162)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_20 = paddle._C_ops.transpose(add__104, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_175 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, full_int_array_175), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(relu__11, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_176 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_164, reshape_165 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_207, full_int_array_176), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__105 = paddle._C_ops.add_(conv2d_64, reshape_164)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_177 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__105, full_int_array_177), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_36 = paddle._C_ops.matmul(reshape__114, reshape__116, False, False)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_36, full_19, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__18 = paddle._C_ops.softmax_(scale__19, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(add_6, [0], full_int_array_178, full_int_array_179, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__106 = paddle._C_ops.add_(softmax__18, slice_18)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_180 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_166, reshape_167 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, full_int_array_180), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_166, add__106, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_181 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_37, full_int_array_181), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(reshape__118, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_182 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_168, reshape_169 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_209, full_int_array_182), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__107 = paddle._C_ops.add_(conv2d_65, reshape_168)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(relu__11, parameter_210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_183 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_170, reshape_171 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_211, full_int_array_183), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__108 = paddle._C_ops.add_(conv2d_66, reshape_170)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_21 = paddle._C_ops.transpose(add__108, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_184 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu__11, parameter_212, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_185 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_172, reshape_173 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_213, full_int_array_185), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__109 = paddle._C_ops.add_(conv2d_67, reshape_172)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_186 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__109, full_int_array_186), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_38 = paddle._C_ops.matmul(reshape__120, reshape__122, False, False)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_38, full_20, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__19 = paddle._C_ops.softmax_(scale__20, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(add_6, [0], full_int_array_187, full_int_array_188, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__110 = paddle._C_ops.add_(softmax__19, slice_19)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_189 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_174, reshape_175 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, full_int_array_189), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_174, add__110, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_190 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_39, full_int_array_190), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(reshape__124, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_191 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_176, reshape_177 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_191), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__111 = paddle._C_ops.add_(conv2d_68, reshape_176)

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__112 = paddle._C_ops.add_(add__111, add__107)

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu__11, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_192 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_178, reshape_179 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_217, full_int_array_192), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__113 = paddle._C_ops.add_(conv2d_69, reshape_178)

        # pd_op.transpose: (1x25x32x175xf32) <- (1x32x175x25xf32)
        transpose_22 = paddle._C_ops.transpose(add__113, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_193 = [1, 25, 5600]

        # pd_op.reshape_: (1x25x5600xf32, 0x1x25x32x175xf32) <- (1x25x32x175xf32, 3xi64)
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, full_int_array_193), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x32x175x25xf32) <- (1x128x175x25xf32, 32x128x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(relu__11, parameter_218, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_194 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_180, reshape_181 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_219, full_int_array_194), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x175x25xf32) <- (1x32x175x25xf32, 1x32x1x1xf32)
        add__114 = paddle._C_ops.add_(conv2d_70, reshape_180)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_195 = [1, 5600, 25]

        # pd_op.reshape_: (1x5600x25xf32, 0x1x32x175x25xf32) <- (1x32x175x25xf32, 3xi64)
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__114, full_int_array_195), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5600xf32, 1x5600x25xf32)
        matmul_40 = paddle._C_ops.matmul(reshape__126, reshape__128, False, False)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('0.000178571'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_40, full_21, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__20 = paddle._C_ops.softmax_(scale__21, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(add_6, [0], full_int_array_196, full_int_array_197, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__115 = paddle._C_ops.add_(softmax__20, slice_20)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_198 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_182, reshape_183 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__11, full_int_array_198), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_182, add__115, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_199 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_41, full_int_array_199), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(reshape__130, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_200 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_184, reshape_185 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_221, full_int_array_200), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__116 = paddle._C_ops.add_(conv2d_71, reshape_184)

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__117 = paddle._C_ops.add_(add__116, add__112)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__117, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__118 = paddle._C_ops.add_(batch_norm__96, relu__11)

        # pd_op.relu_: (1x128x175x25xf32) <- (1x128x175x25xf32)
        relu__12 = paddle._C_ops.relu_(add__118)

        # pd_op.conv2d: (1x128x175x25xf32) <- (1x128x175x25xf32, 128x128x9x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(relu__12, parameter_226, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_201 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_186, reshape_187 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_227, full_int_array_201), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x1x1xf32)
        add__119 = paddle._C_ops.add_(conv2d_72, reshape_186)

        # pd_op.batch_norm_: (1x128x175x25xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x175x25xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__119, parameter_228, parameter_229, parameter_230, parameter_231, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x128x175x25xf32) <- (1x128x175x25xf32, 1x128x175x25xf32)
        add__120 = paddle._C_ops.add_(batch_norm__102, relu__11)

        # pd_op.relu_: (1x128x175x25xf32) <- (1x128x175x25xf32)
        relu__13 = paddle._C_ops.relu_(add__120)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_7 = paddle._C_ops.add(parameter_232, parameter_233)

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(relu__13, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_202 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_188, reshape_189 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_235, full_int_array_202), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__121 = paddle._C_ops.add_(conv2d_73, reshape_188)

        # pd_op.transpose: (1x25x64x175xf32) <- (1x64x175x25xf32)
        transpose_23 = paddle._C_ops.transpose(add__121, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_203 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x64x175xf32) <- (1x25x64x175xf32, 3xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, full_int_array_203), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu__13, parameter_236, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_204 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_190, reshape_191 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_237, full_int_array_204), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__122 = paddle._C_ops.add_(conv2d_74, reshape_190)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_205 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x64x175x25xf32) <- (1x64x175x25xf32, 3xi64)
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__122, full_int_array_205), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_42 = paddle._C_ops.matmul(reshape__132, reshape__134, False, False)

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_42, full_22, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__21 = paddle._C_ops.softmax_(scale__22, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(add_7, [0], full_int_array_206, full_int_array_207, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__123 = paddle._C_ops.add_(softmax__21, slice_21)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_208 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_192, reshape_193 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, full_int_array_208), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_43 = paddle._C_ops.matmul(reshape_192, add__123, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_209 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_43, full_int_array_209), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf32) <- (1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(reshape__136, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_210 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_194, reshape_195 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_239, full_int_array_210), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x1x1xf32)
        add__124 = paddle._C_ops.add_(conv2d_75, reshape_194)

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__13, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_211 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_196, reshape_197 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_241, full_int_array_211), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__125 = paddle._C_ops.add_(conv2d_76, reshape_196)

        # pd_op.transpose: (1x25x64x175xf32) <- (1x64x175x25xf32)
        transpose_24 = paddle._C_ops.transpose(add__125, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_212 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x64x175xf32) <- (1x25x64x175xf32, 3xi64)
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, full_int_array_212), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu__13, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_213 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_198, reshape_199 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_243, full_int_array_213), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__126 = paddle._C_ops.add_(conv2d_77, reshape_198)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_214 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x64x175x25xf32) <- (1x64x175x25xf32, 3xi64)
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__126, full_int_array_214), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_44 = paddle._C_ops.matmul(reshape__138, reshape__140, False, False)

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_44, full_23, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__22 = paddle._C_ops.softmax_(scale__23, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_215 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_216 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(add_7, [0], full_int_array_215, full_int_array_216, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__127 = paddle._C_ops.add_(softmax__22, slice_22)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_217 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_200, reshape_201 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, full_int_array_217), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_200, add__127, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_218 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_45, full_int_array_218), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf32) <- (1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(reshape__142, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_219 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_202, reshape_203 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_245, full_int_array_219), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x1x1xf32)
        add__128 = paddle._C_ops.add_(conv2d_78, reshape_202)

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x175x25xf32)
        add__129 = paddle._C_ops.add_(add__128, add__124)

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu__13, parameter_246, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_220 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_204, reshape_205 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_247, full_int_array_220), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__130 = paddle._C_ops.add_(conv2d_79, reshape_204)

        # pd_op.transpose: (1x25x64x175xf32) <- (1x64x175x25xf32)
        transpose_25 = paddle._C_ops.transpose(add__130, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_221 = [1, 25, 11200]

        # pd_op.reshape_: (1x25x11200xf32, 0x1x25x64x175xf32) <- (1x25x64x175xf32, 3xi64)
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, full_int_array_221), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x175x25xf32) <- (1x128x175x25xf32, 64x128x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu__13, parameter_248, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_222 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_206, reshape_207 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_249, full_int_array_222), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x175x25xf32) <- (1x64x175x25xf32, 1x64x1x1xf32)
        add__131 = paddle._C_ops.add_(conv2d_80, reshape_206)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_223 = [1, 11200, 25]

        # pd_op.reshape_: (1x11200x25xf32, 0x1x64x175x25xf32) <- (1x64x175x25xf32, 3xi64)
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__131, full_int_array_223), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x11200xf32, 1x11200x25xf32)
        matmul_46 = paddle._C_ops.matmul(reshape__144, reshape__146, False, False)

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full([1], float('8.92857e-05'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__24 = paddle._C_ops.scale_(matmul_46, full_24, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__23 = paddle._C_ops.softmax_(scale__24, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_224 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(add_7, [0], full_int_array_224, full_int_array_225, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__132 = paddle._C_ops.add_(softmax__23, slice_23)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_226 = [1, 22400, 25]

        # pd_op.reshape: (1x22400x25xf32, 0x1x128x175x25xf32) <- (1x128x175x25xf32, 3xi64)
        reshape_208, reshape_209 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__13, full_int_array_226), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22400x25xf32) <- (1x22400x25xf32, 1x25x25xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_208, add__132, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_227 = [1, 128, 175, 25]

        # pd_op.reshape_: (1x128x175x25xf32, 0x1x22400x25xf32) <- (1x22400x25xf32, 4xi64)
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_47, full_int_array_227), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x175x25xf32) <- (1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(reshape__148, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_228 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_210, reshape_211 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_251, full_int_array_228), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x1x1xf32)
        add__133 = paddle._C_ops.add_(conv2d_81, reshape_210)

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x175x25xf32)
        add__134 = paddle._C_ops.add_(add__133, add__129)

        # pd_op.batch_norm_: (1x256x175x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x175x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__134, parameter_252, parameter_253, parameter_254, parameter_255, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x256x175x25xf32) <- (1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu__13, parameter_256, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_229 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_212, reshape_213 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_257, full_int_array_229), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x1x1xf32)
        add__135 = paddle._C_ops.add_(conv2d_82, reshape_212)

        # pd_op.batch_norm_: (1x256x175x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x175x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__135, parameter_258, parameter_259, parameter_260, parameter_261, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x175x25xf32) <- (1x256x175x25xf32, 1x256x175x25xf32)
        add__136 = paddle._C_ops.add_(batch_norm__108, batch_norm__114)

        # pd_op.relu_: (1x256x175x25xf32) <- (1x256x175x25xf32)
        relu__14 = paddle._C_ops.relu_(add__136)

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x175x25xf32, 256x256x9x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu__14, parameter_262, [2, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_230 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_214, reshape_215 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_263, full_int_array_230), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__137 = paddle._C_ops.add_(conv2d_83, reshape_214)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__137, parameter_264, parameter_265, parameter_266, parameter_267, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x128x175x25xf32, 256x128x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu__13, parameter_268, [2, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_231 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_216, reshape_217 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_269, full_int_array_231), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__138 = paddle._C_ops.add_(conv2d_84, reshape_216)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__138, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__139 = paddle._C_ops.add_(batch_norm__120, batch_norm__126)

        # pd_op.relu_: (1x256x88x25xf32) <- (1x256x88x25xf32)
        relu__15 = paddle._C_ops.relu_(add__139)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_8 = paddle._C_ops.add(parameter_274, parameter_275)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu__15, parameter_276, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_232 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_218, reshape_219 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_277, full_int_array_232), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__140 = paddle._C_ops.add_(conv2d_85, reshape_218)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_26 = paddle._C_ops.transpose(add__140, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_233 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, full_int_array_233), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(relu__15, parameter_278, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_234 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_220, reshape_221 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_279, full_int_array_234), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__141 = paddle._C_ops.add_(conv2d_86, reshape_220)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_235 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__141, full_int_array_235), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_48 = paddle._C_ops.matmul(reshape__150, reshape__152, False, False)

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__25 = paddle._C_ops.scale_(matmul_48, full_25, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__24 = paddle._C_ops.softmax_(scale__25, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_236 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_237 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(add_8, [0], full_int_array_236, full_int_array_237, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__142 = paddle._C_ops.add_(softmax__24, slice_24)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_238 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_222, reshape_223 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, full_int_array_238), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_49 = paddle._C_ops.matmul(reshape_222, add__142, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_239 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_49, full_int_array_239), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(reshape__154, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_240 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_224, reshape_225 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_281, full_int_array_240), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__143 = paddle._C_ops.add_(conv2d_87, reshape_224)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(relu__15, parameter_282, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_241 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_226, reshape_227 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_283, full_int_array_241), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__144 = paddle._C_ops.add_(conv2d_88, reshape_226)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_27 = paddle._C_ops.transpose(add__144, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_242 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, full_int_array_242), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_89 = paddle._C_ops.conv2d(relu__15, parameter_284, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_243 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_228, reshape_229 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_285, full_int_array_243), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__145 = paddle._C_ops.add_(conv2d_89, reshape_228)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_244 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__145, full_int_array_244), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_50 = paddle._C_ops.matmul(reshape__156, reshape__158, False, False)

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__26 = paddle._C_ops.scale_(matmul_50, full_26, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__25 = paddle._C_ops.softmax_(scale__26, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_245 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_246 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(add_8, [0], full_int_array_245, full_int_array_246, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__146 = paddle._C_ops.add_(softmax__25, slice_25)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_247 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_230, reshape_231 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, full_int_array_247), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_51 = paddle._C_ops.matmul(reshape_230, add__146, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_248 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_51, full_int_array_248), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(reshape__160, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_249 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_232, reshape_233 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_287, full_int_array_249), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__147 = paddle._C_ops.add_(conv2d_90, reshape_232)

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__148 = paddle._C_ops.add_(add__147, add__143)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(relu__15, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_250 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_234, reshape_235 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_289, full_int_array_250), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__149 = paddle._C_ops.add_(conv2d_91, reshape_234)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_28 = paddle._C_ops.transpose(add__149, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_251 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, full_int_array_251), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_92 = paddle._C_ops.conv2d(relu__15, parameter_290, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_252 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_236, reshape_237 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_291, full_int_array_252), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__150 = paddle._C_ops.add_(conv2d_92, reshape_236)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_253 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__150, full_int_array_253), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_52 = paddle._C_ops.matmul(reshape__162, reshape__164, False, False)

        # pd_op.full: (1xf32) <- ()
        full_27 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__27 = paddle._C_ops.scale_(matmul_52, full_27, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__26 = paddle._C_ops.softmax_(scale__27, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_254 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_255 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(add_8, [0], full_int_array_254, full_int_array_255, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__151 = paddle._C_ops.add_(softmax__26, slice_26)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_256 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_238, reshape_239 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__15, full_int_array_256), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_238, add__151, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_257 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_53, full_int_array_257), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(reshape__166, parameter_292, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_258 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_240, reshape_241 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_293, full_int_array_258), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__152 = paddle._C_ops.add_(conv2d_93, reshape_240)

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__153 = paddle._C_ops.add_(add__152, add__148)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__153, parameter_294, parameter_295, parameter_296, parameter_297, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__154 = paddle._C_ops.add_(batch_norm__132, relu__15)

        # pd_op.relu_: (1x256x88x25xf32) <- (1x256x88x25xf32)
        relu__16 = paddle._C_ops.relu_(add__154)

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x9x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu__16, parameter_298, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_259 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_242, reshape_243 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_299, full_int_array_259), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__155 = paddle._C_ops.add_(conv2d_94, reshape_242)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__155, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__156 = paddle._C_ops.add_(batch_norm__138, relu__15)

        # pd_op.relu_: (1x256x88x25xf32) <- (1x256x88x25xf32)
        relu__17 = paddle._C_ops.relu_(add__156)

        # pd_op.add: (3x25x25xf32) <- (3x25x25xf32, 3x25x25xf32)
        add_9 = paddle._C_ops.add(parameter_304, parameter_305)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_95 = paddle._C_ops.conv2d(relu__17, parameter_306, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_260 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_244, reshape_245 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_307, full_int_array_260), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__157 = paddle._C_ops.add_(conv2d_95, reshape_244)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_29 = paddle._C_ops.transpose(add__157, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_261 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, full_int_array_261), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu__17, parameter_308, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_262 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_246, reshape_247 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_309, full_int_array_262), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__158 = paddle._C_ops.add_(conv2d_96, reshape_246)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_263 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__158, full_int_array_263), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_54 = paddle._C_ops.matmul(reshape__168, reshape__170, False, False)

        # pd_op.full: (1xf32) <- ()
        full_28 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__28 = paddle._C_ops.scale_(matmul_54, full_28, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__27 = paddle._C_ops.softmax_(scale__28, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_264 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_265 = [1]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(add_9, [0], full_int_array_264, full_int_array_265, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__159 = paddle._C_ops.add_(softmax__27, slice_27)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_266 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_248, reshape_249 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, full_int_array_266), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_55 = paddle._C_ops.matmul(reshape_248, add__159, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_267 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_55, full_int_array_267), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_97 = paddle._C_ops.conv2d(reshape__172, parameter_310, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_268 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_250, reshape_251 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_311, full_int_array_268), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__160 = paddle._C_ops.add_(conv2d_97, reshape_250)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_98 = paddle._C_ops.conv2d(relu__17, parameter_312, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_269 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_252, reshape_253 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_313, full_int_array_269), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__161 = paddle._C_ops.add_(conv2d_98, reshape_252)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_30 = paddle._C_ops.transpose(add__161, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_270 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, full_int_array_270), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_99 = paddle._C_ops.conv2d(relu__17, parameter_314, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_271 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_254, reshape_255 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_315, full_int_array_271), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__162 = paddle._C_ops.add_(conv2d_99, reshape_254)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_272 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__162, full_int_array_272), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_56 = paddle._C_ops.matmul(reshape__174, reshape__176, False, False)

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__29 = paddle._C_ops.scale_(matmul_56, full_29, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__28 = paddle._C_ops.softmax_(scale__29, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_273 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_274 = [2]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(add_9, [0], full_int_array_273, full_int_array_274, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__163 = paddle._C_ops.add_(softmax__28, slice_28)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_275 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_256, reshape_257 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, full_int_array_275), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_256, add__163, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_276 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_57, full_int_array_276), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_100 = paddle._C_ops.conv2d(reshape__178, parameter_316, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_277 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_258, reshape_259 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_317, full_int_array_277), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__164 = paddle._C_ops.add_(conv2d_100, reshape_258)

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__165 = paddle._C_ops.add_(add__164, add__160)

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu__17, parameter_318, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_278 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_260, reshape_261 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_319, full_int_array_278), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__166 = paddle._C_ops.add_(conv2d_101, reshape_260)

        # pd_op.transpose: (1x25x64x88xf32) <- (1x64x88x25xf32)
        transpose_31 = paddle._C_ops.transpose(add__166, [0, 3, 1, 2])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_279 = [1, 25, 5632]

        # pd_op.reshape_: (1x25x5632xf32, 0x1x25x64x88xf32) <- (1x25x64x88xf32, 3xi64)
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_31, full_int_array_279), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x64x88x25xf32) <- (1x256x88x25xf32, 64x256x1x1xf32)
        conv2d_102 = paddle._C_ops.conv2d(relu__17, parameter_320, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_280 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_262, reshape_263 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_321, full_int_array_280), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x64x88x25xf32) <- (1x64x88x25xf32, 1x64x1x1xf32)
        add__167 = paddle._C_ops.add_(conv2d_102, reshape_262)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_281 = [1, 5632, 25]

        # pd_op.reshape_: (1x5632x25xf32, 0x1x64x88x25xf32) <- (1x64x88x25xf32, 3xi64)
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__167, full_int_array_281), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x25x25xf32) <- (1x25x5632xf32, 1x5632x25xf32)
        matmul_58 = paddle._C_ops.matmul(reshape__180, reshape__182, False, False)

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full([1], float('0.000177557'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x25x25xf32) <- (1x25x25xf32, 1xf32)
        scale__30 = paddle._C_ops.scale_(matmul_58, full_30, float('0'), True)

        # pd_op.softmax_: (1x25x25xf32) <- (1x25x25xf32)
        softmax__29 = paddle._C_ops.softmax_(scale__30, -2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_282 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_283 = [3]

        # pd_op.slice: (25x25xf32) <- (3x25x25xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(add_9, [0], full_int_array_282, full_int_array_283, [1], [0])

        # pd_op.add_: (1x25x25xf32) <- (1x25x25xf32, 25x25xf32)
        add__168 = paddle._C_ops.add_(softmax__29, slice_29)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_284 = [1, 22528, 25]

        # pd_op.reshape: (1x22528x25xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 3xi64)
        reshape_264, reshape_265 = (lambda x, f: f(x))(paddle._C_ops.reshape(relu__17, full_int_array_284), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (1x22528x25xf32) <- (1x22528x25xf32, 1x25x25xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_264, add__168, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_285 = [1, 256, 88, 25]

        # pd_op.reshape_: (1x256x88x25xf32, 0x1x22528x25xf32) <- (1x22528x25xf32, 4xi64)
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_59, full_int_array_285), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x1x1xf32)
        conv2d_103 = paddle._C_ops.conv2d(reshape__184, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_286 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_266, reshape_267 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_323, full_int_array_286), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__169 = paddle._C_ops.add_(conv2d_103, reshape_266)

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__170 = paddle._C_ops.add_(add__169, add__165)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__170, parameter_324, parameter_325, parameter_326, parameter_327, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__171 = paddle._C_ops.add_(batch_norm__144, relu__17)

        # pd_op.relu_: (1x256x88x25xf32) <- (1x256x88x25xf32)
        relu__18 = paddle._C_ops.relu_(add__171)

        # pd_op.conv2d: (1x256x88x25xf32) <- (1x256x88x25xf32, 256x256x9x1xf32)
        conv2d_104 = paddle._C_ops.conv2d(relu__18, parameter_328, [1, 1], [4, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_287 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_268, reshape_269 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_329, full_int_array_287), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x1x1xf32)
        add__172 = paddle._C_ops.add_(conv2d_104, reshape_268)

        # pd_op.batch_norm_: (1x256x88x25xf32, 256xf32, 256xf32, xf32, xf32, None) <- (1x256x88x25xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__172, parameter_330, parameter_331, parameter_332, parameter_333, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x256x88x25xf32) <- (1x256x88x25xf32, 1x256x88x25xf32)
        add__173 = paddle._C_ops.add_(batch_norm__150, relu__17)

        # pd_op.relu_: (1x256x88x25xf32) <- (1x256x88x25xf32)
        relu__19 = paddle._C_ops.relu_(add__173)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_288 = [1, 1, 256, -1]

        # pd_op.reshape_: (1x1x256x2200xf32, 0x1x256x88x25xf32) <- (1x256x88x25xf32, 4xi64)
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(relu__19, full_int_array_288), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.mean: (1x1x256xf32) <- (1x1x256x2200xf32)
        mean_0 = paddle._C_ops.mean(reshape__186, [3], False)

        # pd_op.mean: (1x256xf32) <- (1x1x256xf32)
        mean_1 = paddle._C_ops.mean(mean_0, [1], False)

        # pd_op.matmul: (1x60xf32) <- (1x256xf32, 256x60xf32)
        matmul_60 = paddle._C_ops.matmul(mean_1, parameter_334, False, False)

        # pd_op.add_: (1x60xf32) <- (1x60xf32, 60xf32)
        add__174 = paddle._C_ops.add_(matmul_60, parameter_335)
        return add__174



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

    def forward(self, parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_305, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_335, feed_0):
        return self.builtin_module_1414_0_0(parameter_3, parameter_0, parameter_2, parameter_1, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_27, parameter_24, parameter_26, parameter_25, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_165, parameter_162, parameter_164, parameter_163, parameter_166, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, parameter_248, parameter_249, parameter_250, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_279, parameter_280, parameter_281, parameter_282, parameter_283, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_305, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_311, parameter_312, parameter_313, parameter_314, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_323, parameter_327, parameter_324, parameter_326, parameter_325, parameter_328, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_335, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1414_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_3
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([16, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64, 2, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64, 64, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128, 128, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([3, 25, 25], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([256, 256, 9, 1], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256, 60], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 2, 350, 25, 1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_3
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[16, 2, 1, 1], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64, 2, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[64, 64, 9, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[128, 128, 9, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[3, 25, 25], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[256, 256, 9, 1], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[256, 60], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 2, 350, 25, 1], dtype='float32'),
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