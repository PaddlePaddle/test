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
    return [4440][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_4725_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_212, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_234, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, feed_0):

        # pd_op.conv2d: (-1x16x112x112xf32) <- (-1x3x224x224xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x112x112xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x112x112xf32) <- (-1x16x112x112xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x16x112x112xf32, 32x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(hardswish_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x32x56x56xf32) <- (-1x32x56x56xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(hardswish_1, parameter_10, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x64x28x28xf32) <- (-1x64x28x28xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(hardswish_2, parameter_15, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.flatten_: (-1x128x196xf32, None) <- (-1x128x14x14xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(batch_norm__18, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x128xf32) <- (-1x128x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten__0, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_0 = paddle._C_ops.shape(transpose_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_0, parameter_20, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_0, flatten_1 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_0, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_0, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_1 = paddle._C_ops.shape(matmul_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_1, full_0, full_1]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__24, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, full_2, full_3, full_4]

        # pd_op.reshape_: (-1x196x4x64xf32, 0x-1x196x256xf32) <- (-1x196x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32]) <- (-1x196x4x64xf32, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape__2, full_int_array_4, full_5)

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_2 = split_0[0]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_1 = paddle._C_ops.transpose(slice_2, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_3 = split_0[1]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x32xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_4 = split_0[2]

        # pd_op.transpose: (-1x4x196x32xf32) <- (-1x196x4x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_4, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x16x196xf32) <- (-1x4x196x16xf32)
        transpose_4 = paddle._C_ops.transpose(transpose_2, [0, 1, 3, 2])

        # pd_op.transpose: (196x4xf32) <- (4x196xf32)
        transpose_5 = paddle._C_ops.transpose(parameter_25, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(parameter_26, [0], full_int_array_5, full_int_array_6, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(transpose_5, slice_5, full_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(parameter_26, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(transpose_5, slice_6, full_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(parameter_26, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(transpose_5, slice_7, full_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [4]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(parameter_26, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(transpose_5, slice_8, full_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [5]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(parameter_26, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(transpose_5, slice_9, full_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [6]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(parameter_26, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(transpose_5, slice_10, full_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [7]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(parameter_26, [0], full_int_array_17, full_int_array_18, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(transpose_5, slice_11, full_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [8]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(parameter_26, [0], full_int_array_19, full_int_array_20, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(transpose_5, slice_12, full_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [9]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(parameter_26, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(transpose_5, slice_13, full_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [10]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(parameter_26, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(transpose_5, slice_14, full_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [11]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(parameter_26, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(transpose_5, slice_15, full_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [12]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(parameter_26, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(transpose_5, slice_16, full_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [13]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(parameter_26, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(transpose_5, slice_17, full_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [14]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(parameter_26, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(transpose_5, slice_18, full_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [15]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(parameter_26, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(transpose_5, slice_19, full_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [16]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(parameter_26, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(transpose_5, slice_20, full_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [17]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(parameter_26, [0], full_int_array_37, full_int_array_38, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_16 = paddle._C_ops.gather(transpose_5, slice_21, full_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [18]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(parameter_26, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_17 = paddle._C_ops.gather(transpose_5, slice_22, full_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [19]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(parameter_26, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_18 = paddle._C_ops.gather(transpose_5, slice_23, full_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [20]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(parameter_26, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_19 = paddle._C_ops.gather(transpose_5, slice_24, full_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [21]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(parameter_26, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_20 = paddle._C_ops.gather(transpose_5, slice_25, full_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [22]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(parameter_26, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_21 = paddle._C_ops.gather(transpose_5, slice_26, full_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [23]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(parameter_26, [0], full_int_array_49, full_int_array_50, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_22 = paddle._C_ops.gather(transpose_5, slice_27, full_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [24]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(parameter_26, [0], full_int_array_51, full_int_array_52, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_23 = paddle._C_ops.gather(transpose_5, slice_28, full_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [25]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(parameter_26, [0], full_int_array_53, full_int_array_54, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_24 = paddle._C_ops.gather(transpose_5, slice_29, full_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [26]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(parameter_26, [0], full_int_array_55, full_int_array_56, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_25 = paddle._C_ops.gather(transpose_5, slice_30, full_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [27]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(parameter_26, [0], full_int_array_57, full_int_array_58, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_26 = paddle._C_ops.gather(transpose_5, slice_31, full_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [28]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(parameter_26, [0], full_int_array_59, full_int_array_60, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_27 = paddle._C_ops.gather(transpose_5, slice_32, full_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [29]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(parameter_26, [0], full_int_array_61, full_int_array_62, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_28 = paddle._C_ops.gather(transpose_5, slice_33, full_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [30]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(parameter_26, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_29 = paddle._C_ops.gather(transpose_5, slice_34, full_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [31]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(parameter_26, [0], full_int_array_65, full_int_array_66, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_30 = paddle._C_ops.gather(transpose_5, slice_35, full_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [32]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(parameter_26, [0], full_int_array_67, full_int_array_68, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_31 = paddle._C_ops.gather(transpose_5, slice_36, full_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [33]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(parameter_26, [0], full_int_array_69, full_int_array_70, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_32 = paddle._C_ops.gather(transpose_5, slice_37, full_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [34]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(parameter_26, [0], full_int_array_71, full_int_array_72, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_33 = paddle._C_ops.gather(transpose_5, slice_38, full_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [35]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(parameter_26, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_34 = paddle._C_ops.gather(transpose_5, slice_39, full_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [36]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(parameter_26, [0], full_int_array_75, full_int_array_76, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_35 = paddle._C_ops.gather(transpose_5, slice_40, full_41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [37]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(parameter_26, [0], full_int_array_77, full_int_array_78, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_36 = paddle._C_ops.gather(transpose_5, slice_41, full_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [38]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(parameter_26, [0], full_int_array_79, full_int_array_80, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_37 = paddle._C_ops.gather(transpose_5, slice_42, full_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [39]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(parameter_26, [0], full_int_array_81, full_int_array_82, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_38 = paddle._C_ops.gather(transpose_5, slice_43, full_44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [40]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(parameter_26, [0], full_int_array_83, full_int_array_84, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_39 = paddle._C_ops.gather(transpose_5, slice_44, full_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [41]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(parameter_26, [0], full_int_array_85, full_int_array_86, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_40 = paddle._C_ops.gather(transpose_5, slice_45, full_46)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [42]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(parameter_26, [0], full_int_array_87, full_int_array_88, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_41 = paddle._C_ops.gather(transpose_5, slice_46, full_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [43]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(parameter_26, [0], full_int_array_89, full_int_array_90, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_42 = paddle._C_ops.gather(transpose_5, slice_47, full_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [44]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(parameter_26, [0], full_int_array_91, full_int_array_92, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_43 = paddle._C_ops.gather(transpose_5, slice_48, full_49)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [45]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(parameter_26, [0], full_int_array_93, full_int_array_94, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_44 = paddle._C_ops.gather(transpose_5, slice_49, full_50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [46]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(parameter_26, [0], full_int_array_95, full_int_array_96, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_45 = paddle._C_ops.gather(transpose_5, slice_50, full_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [47]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(parameter_26, [0], full_int_array_97, full_int_array_98, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_46 = paddle._C_ops.gather(transpose_5, slice_51, full_52)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [48]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(parameter_26, [0], full_int_array_99, full_int_array_100, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_47 = paddle._C_ops.gather(transpose_5, slice_52, full_53)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [49]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(parameter_26, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_48 = paddle._C_ops.gather(transpose_5, slice_53, full_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [49]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [50]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(parameter_26, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_49 = paddle._C_ops.gather(transpose_5, slice_54, full_55)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [50]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [51]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(parameter_26, [0], full_int_array_105, full_int_array_106, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_50 = paddle._C_ops.gather(transpose_5, slice_55, full_56)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [51]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [52]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(parameter_26, [0], full_int_array_107, full_int_array_108, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_51 = paddle._C_ops.gather(transpose_5, slice_56, full_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [52]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [53]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(parameter_26, [0], full_int_array_109, full_int_array_110, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_52 = paddle._C_ops.gather(transpose_5, slice_57, full_58)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [53]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [54]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(parameter_26, [0], full_int_array_111, full_int_array_112, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_53 = paddle._C_ops.gather(transpose_5, slice_58, full_59)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [54]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [55]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(parameter_26, [0], full_int_array_113, full_int_array_114, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_54 = paddle._C_ops.gather(transpose_5, slice_59, full_60)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [55]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [56]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(parameter_26, [0], full_int_array_115, full_int_array_116, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_55 = paddle._C_ops.gather(transpose_5, slice_60, full_61)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [56]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [57]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(parameter_26, [0], full_int_array_117, full_int_array_118, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_56 = paddle._C_ops.gather(transpose_5, slice_61, full_62)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [57]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [58]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(parameter_26, [0], full_int_array_119, full_int_array_120, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_57 = paddle._C_ops.gather(transpose_5, slice_62, full_63)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [58]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [59]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(parameter_26, [0], full_int_array_121, full_int_array_122, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_58 = paddle._C_ops.gather(transpose_5, slice_63, full_64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [59]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [60]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(parameter_26, [0], full_int_array_123, full_int_array_124, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_59 = paddle._C_ops.gather(transpose_5, slice_64, full_65)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [60]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [61]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(parameter_26, [0], full_int_array_125, full_int_array_126, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_60 = paddle._C_ops.gather(transpose_5, slice_65, full_66)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [61]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [62]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(parameter_26, [0], full_int_array_127, full_int_array_128, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_61 = paddle._C_ops.gather(transpose_5, slice_66, full_67)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [62]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [63]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(parameter_26, [0], full_int_array_129, full_int_array_130, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_62 = paddle._C_ops.gather(transpose_5, slice_67, full_68)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [63]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [64]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(parameter_26, [0], full_int_array_131, full_int_array_132, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_63 = paddle._C_ops.gather(transpose_5, slice_68, full_69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [64]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [65]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(parameter_26, [0], full_int_array_133, full_int_array_134, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_64 = paddle._C_ops.gather(transpose_5, slice_69, full_70)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [65]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [66]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(parameter_26, [0], full_int_array_135, full_int_array_136, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_65 = paddle._C_ops.gather(transpose_5, slice_70, full_71)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [66]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [67]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(parameter_26, [0], full_int_array_137, full_int_array_138, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_66 = paddle._C_ops.gather(transpose_5, slice_71, full_72)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [67]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [68]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(parameter_26, [0], full_int_array_139, full_int_array_140, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_67 = paddle._C_ops.gather(transpose_5, slice_72, full_73)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [68]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [69]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(parameter_26, [0], full_int_array_141, full_int_array_142, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_68 = paddle._C_ops.gather(transpose_5, slice_73, full_74)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [69]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [70]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(parameter_26, [0], full_int_array_143, full_int_array_144, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_69 = paddle._C_ops.gather(transpose_5, slice_74, full_75)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [70]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [71]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(parameter_26, [0], full_int_array_145, full_int_array_146, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_70 = paddle._C_ops.gather(transpose_5, slice_75, full_76)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [71]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [72]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(parameter_26, [0], full_int_array_147, full_int_array_148, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_71 = paddle._C_ops.gather(transpose_5, slice_76, full_77)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [72]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [73]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(parameter_26, [0], full_int_array_149, full_int_array_150, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_72 = paddle._C_ops.gather(transpose_5, slice_77, full_78)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [73]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [74]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(parameter_26, [0], full_int_array_151, full_int_array_152, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_73 = paddle._C_ops.gather(transpose_5, slice_78, full_79)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [74]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [75]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(parameter_26, [0], full_int_array_153, full_int_array_154, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_74 = paddle._C_ops.gather(transpose_5, slice_79, full_80)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [75]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [76]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(parameter_26, [0], full_int_array_155, full_int_array_156, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_75 = paddle._C_ops.gather(transpose_5, slice_80, full_81)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [76]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [77]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(parameter_26, [0], full_int_array_157, full_int_array_158, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_76 = paddle._C_ops.gather(transpose_5, slice_81, full_82)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [77]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [78]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(parameter_26, [0], full_int_array_159, full_int_array_160, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_77 = paddle._C_ops.gather(transpose_5, slice_82, full_83)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [78]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [79]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(parameter_26, [0], full_int_array_161, full_int_array_162, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_78 = paddle._C_ops.gather(transpose_5, slice_83, full_84)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [79]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [80]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(parameter_26, [0], full_int_array_163, full_int_array_164, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_79 = paddle._C_ops.gather(transpose_5, slice_84, full_85)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [80]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [81]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(parameter_26, [0], full_int_array_165, full_int_array_166, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_80 = paddle._C_ops.gather(transpose_5, slice_85, full_86)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [81]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [82]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(parameter_26, [0], full_int_array_167, full_int_array_168, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_81 = paddle._C_ops.gather(transpose_5, slice_86, full_87)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [82]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [83]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(parameter_26, [0], full_int_array_169, full_int_array_170, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_82 = paddle._C_ops.gather(transpose_5, slice_87, full_88)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [83]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [84]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(parameter_26, [0], full_int_array_171, full_int_array_172, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_83 = paddle._C_ops.gather(transpose_5, slice_88, full_89)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [84]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [85]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(parameter_26, [0], full_int_array_173, full_int_array_174, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_84 = paddle._C_ops.gather(transpose_5, slice_89, full_90)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [85]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [86]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(parameter_26, [0], full_int_array_175, full_int_array_176, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_85 = paddle._C_ops.gather(transpose_5, slice_90, full_91)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [86]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [87]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(parameter_26, [0], full_int_array_177, full_int_array_178, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_86 = paddle._C_ops.gather(transpose_5, slice_91, full_92)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [87]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [88]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(parameter_26, [0], full_int_array_179, full_int_array_180, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_87 = paddle._C_ops.gather(transpose_5, slice_92, full_93)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [88]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [89]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(parameter_26, [0], full_int_array_181, full_int_array_182, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_88 = paddle._C_ops.gather(transpose_5, slice_93, full_94)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [89]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_184 = [90]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(parameter_26, [0], full_int_array_183, full_int_array_184, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_89 = paddle._C_ops.gather(transpose_5, slice_94, full_95)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [90]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [91]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(parameter_26, [0], full_int_array_185, full_int_array_186, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_90 = paddle._C_ops.gather(transpose_5, slice_95, full_96)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [91]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [92]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(parameter_26, [0], full_int_array_187, full_int_array_188, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_91 = paddle._C_ops.gather(transpose_5, slice_96, full_97)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [92]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [93]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(parameter_26, [0], full_int_array_189, full_int_array_190, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_92 = paddle._C_ops.gather(transpose_5, slice_97, full_98)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [93]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [94]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(parameter_26, [0], full_int_array_191, full_int_array_192, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_93 = paddle._C_ops.gather(transpose_5, slice_98, full_99)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [94]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [95]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(parameter_26, [0], full_int_array_193, full_int_array_194, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_94 = paddle._C_ops.gather(transpose_5, slice_99, full_100)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [95]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [96]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(parameter_26, [0], full_int_array_195, full_int_array_196, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_95 = paddle._C_ops.gather(transpose_5, slice_100, full_101)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [96]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_198 = [97]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(parameter_26, [0], full_int_array_197, full_int_array_198, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_96 = paddle._C_ops.gather(transpose_5, slice_101, full_102)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [97]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [98]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(parameter_26, [0], full_int_array_199, full_int_array_200, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_97 = paddle._C_ops.gather(transpose_5, slice_102, full_103)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_201 = [98]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [99]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(parameter_26, [0], full_int_array_201, full_int_array_202, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_98 = paddle._C_ops.gather(transpose_5, slice_103, full_104)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [99]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_204 = [100]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(parameter_26, [0], full_int_array_203, full_int_array_204, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_99 = paddle._C_ops.gather(transpose_5, slice_104, full_105)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [100]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [101]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(parameter_26, [0], full_int_array_205, full_int_array_206, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_100 = paddle._C_ops.gather(transpose_5, slice_105, full_106)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [101]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [102]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(parameter_26, [0], full_int_array_207, full_int_array_208, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_101 = paddle._C_ops.gather(transpose_5, slice_106, full_107)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [102]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [103]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(parameter_26, [0], full_int_array_209, full_int_array_210, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_102 = paddle._C_ops.gather(transpose_5, slice_107, full_108)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_211 = [103]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_212 = [104]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(parameter_26, [0], full_int_array_211, full_int_array_212, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_103 = paddle._C_ops.gather(transpose_5, slice_108, full_109)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_213 = [104]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_214 = [105]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(parameter_26, [0], full_int_array_213, full_int_array_214, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_104 = paddle._C_ops.gather(transpose_5, slice_109, full_110)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_215 = [105]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_216 = [106]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(parameter_26, [0], full_int_array_215, full_int_array_216, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_105 = paddle._C_ops.gather(transpose_5, slice_110, full_111)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_217 = [106]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_218 = [107]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(parameter_26, [0], full_int_array_217, full_int_array_218, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_106 = paddle._C_ops.gather(transpose_5, slice_111, full_112)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_219 = [107]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_220 = [108]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(parameter_26, [0], full_int_array_219, full_int_array_220, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_107 = paddle._C_ops.gather(transpose_5, slice_112, full_113)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_221 = [108]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_222 = [109]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(parameter_26, [0], full_int_array_221, full_int_array_222, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_108 = paddle._C_ops.gather(transpose_5, slice_113, full_114)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_223 = [109]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_224 = [110]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(parameter_26, [0], full_int_array_223, full_int_array_224, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_109 = paddle._C_ops.gather(transpose_5, slice_114, full_115)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [110]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [111]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(parameter_26, [0], full_int_array_225, full_int_array_226, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_110 = paddle._C_ops.gather(transpose_5, slice_115, full_116)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [111]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_228 = [112]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(parameter_26, [0], full_int_array_227, full_int_array_228, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_111 = paddle._C_ops.gather(transpose_5, slice_116, full_117)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_229 = [112]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_230 = [113]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(parameter_26, [0], full_int_array_229, full_int_array_230, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_112 = paddle._C_ops.gather(transpose_5, slice_117, full_118)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_231 = [113]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_232 = [114]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(parameter_26, [0], full_int_array_231, full_int_array_232, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_113 = paddle._C_ops.gather(transpose_5, slice_118, full_119)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_233 = [114]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_234 = [115]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(parameter_26, [0], full_int_array_233, full_int_array_234, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_114 = paddle._C_ops.gather(transpose_5, slice_119, full_120)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_235 = [115]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_236 = [116]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(parameter_26, [0], full_int_array_235, full_int_array_236, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_115 = paddle._C_ops.gather(transpose_5, slice_120, full_121)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_237 = [116]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_238 = [117]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(parameter_26, [0], full_int_array_237, full_int_array_238, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_116 = paddle._C_ops.gather(transpose_5, slice_121, full_122)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_239 = [117]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_240 = [118]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(parameter_26, [0], full_int_array_239, full_int_array_240, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_117 = paddle._C_ops.gather(transpose_5, slice_122, full_123)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_241 = [118]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_242 = [119]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(parameter_26, [0], full_int_array_241, full_int_array_242, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_118 = paddle._C_ops.gather(transpose_5, slice_123, full_124)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_243 = [119]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_244 = [120]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(parameter_26, [0], full_int_array_243, full_int_array_244, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_119 = paddle._C_ops.gather(transpose_5, slice_124, full_125)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_245 = [120]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_246 = [121]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(parameter_26, [0], full_int_array_245, full_int_array_246, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_120 = paddle._C_ops.gather(transpose_5, slice_125, full_126)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_247 = [121]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_248 = [122]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(parameter_26, [0], full_int_array_247, full_int_array_248, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_121 = paddle._C_ops.gather(transpose_5, slice_126, full_127)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_249 = [122]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_250 = [123]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(parameter_26, [0], full_int_array_249, full_int_array_250, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_122 = paddle._C_ops.gather(transpose_5, slice_127, full_128)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_251 = [123]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_252 = [124]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(parameter_26, [0], full_int_array_251, full_int_array_252, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_123 = paddle._C_ops.gather(transpose_5, slice_128, full_129)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_253 = [124]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_254 = [125]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(parameter_26, [0], full_int_array_253, full_int_array_254, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_124 = paddle._C_ops.gather(transpose_5, slice_129, full_130)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_255 = [125]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_256 = [126]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(parameter_26, [0], full_int_array_255, full_int_array_256, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_125 = paddle._C_ops.gather(transpose_5, slice_130, full_131)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_257 = [126]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_258 = [127]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(parameter_26, [0], full_int_array_257, full_int_array_258, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_126 = paddle._C_ops.gather(transpose_5, slice_131, full_132)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_259 = [127]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_260 = [128]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(parameter_26, [0], full_int_array_259, full_int_array_260, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_127 = paddle._C_ops.gather(transpose_5, slice_132, full_133)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_261 = [128]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_262 = [129]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(parameter_26, [0], full_int_array_261, full_int_array_262, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_128 = paddle._C_ops.gather(transpose_5, slice_133, full_134)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_263 = [129]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_264 = [130]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(parameter_26, [0], full_int_array_263, full_int_array_264, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_129 = paddle._C_ops.gather(transpose_5, slice_134, full_135)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_265 = [130]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_266 = [131]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(parameter_26, [0], full_int_array_265, full_int_array_266, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_130 = paddle._C_ops.gather(transpose_5, slice_135, full_136)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_267 = [131]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_268 = [132]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(parameter_26, [0], full_int_array_267, full_int_array_268, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_131 = paddle._C_ops.gather(transpose_5, slice_136, full_137)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_269 = [132]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_270 = [133]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(parameter_26, [0], full_int_array_269, full_int_array_270, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_132 = paddle._C_ops.gather(transpose_5, slice_137, full_138)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_271 = [133]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_272 = [134]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(parameter_26, [0], full_int_array_271, full_int_array_272, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_133 = paddle._C_ops.gather(transpose_5, slice_138, full_139)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_273 = [134]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_274 = [135]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(parameter_26, [0], full_int_array_273, full_int_array_274, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_134 = paddle._C_ops.gather(transpose_5, slice_139, full_140)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_275 = [135]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_276 = [136]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(parameter_26, [0], full_int_array_275, full_int_array_276, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_135 = paddle._C_ops.gather(transpose_5, slice_140, full_141)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_277 = [136]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_278 = [137]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(parameter_26, [0], full_int_array_277, full_int_array_278, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_136 = paddle._C_ops.gather(transpose_5, slice_141, full_142)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_279 = [137]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_280 = [138]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(parameter_26, [0], full_int_array_279, full_int_array_280, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_137 = paddle._C_ops.gather(transpose_5, slice_142, full_143)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_281 = [138]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_282 = [139]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(parameter_26, [0], full_int_array_281, full_int_array_282, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_138 = paddle._C_ops.gather(transpose_5, slice_143, full_144)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_283 = [139]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_284 = [140]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(parameter_26, [0], full_int_array_283, full_int_array_284, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_139 = paddle._C_ops.gather(transpose_5, slice_144, full_145)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_285 = [140]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_286 = [141]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(parameter_26, [0], full_int_array_285, full_int_array_286, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_140 = paddle._C_ops.gather(transpose_5, slice_145, full_146)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_287 = [141]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_288 = [142]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(parameter_26, [0], full_int_array_287, full_int_array_288, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_141 = paddle._C_ops.gather(transpose_5, slice_146, full_147)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_289 = [142]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_290 = [143]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(parameter_26, [0], full_int_array_289, full_int_array_290, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_142 = paddle._C_ops.gather(transpose_5, slice_147, full_148)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_291 = [143]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_292 = [144]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_148 = paddle._C_ops.slice(parameter_26, [0], full_int_array_291, full_int_array_292, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_143 = paddle._C_ops.gather(transpose_5, slice_148, full_149)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_293 = [144]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_294 = [145]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_149 = paddle._C_ops.slice(parameter_26, [0], full_int_array_293, full_int_array_294, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_144 = paddle._C_ops.gather(transpose_5, slice_149, full_150)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_295 = [145]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_296 = [146]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(parameter_26, [0], full_int_array_295, full_int_array_296, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_145 = paddle._C_ops.gather(transpose_5, slice_150, full_151)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_297 = [146]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_298 = [147]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_151 = paddle._C_ops.slice(parameter_26, [0], full_int_array_297, full_int_array_298, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_146 = paddle._C_ops.gather(transpose_5, slice_151, full_152)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_299 = [147]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_300 = [148]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(parameter_26, [0], full_int_array_299, full_int_array_300, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_147 = paddle._C_ops.gather(transpose_5, slice_152, full_153)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_301 = [148]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_302 = [149]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(parameter_26, [0], full_int_array_301, full_int_array_302, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_148 = paddle._C_ops.gather(transpose_5, slice_153, full_154)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_303 = [149]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_304 = [150]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(parameter_26, [0], full_int_array_303, full_int_array_304, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_149 = paddle._C_ops.gather(transpose_5, slice_154, full_155)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_305 = [150]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_306 = [151]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(parameter_26, [0], full_int_array_305, full_int_array_306, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_150 = paddle._C_ops.gather(transpose_5, slice_155, full_156)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_307 = [151]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_308 = [152]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_156 = paddle._C_ops.slice(parameter_26, [0], full_int_array_307, full_int_array_308, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_151 = paddle._C_ops.gather(transpose_5, slice_156, full_157)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_309 = [152]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_310 = [153]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(parameter_26, [0], full_int_array_309, full_int_array_310, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_152 = paddle._C_ops.gather(transpose_5, slice_157, full_158)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_311 = [153]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_312 = [154]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_158 = paddle._C_ops.slice(parameter_26, [0], full_int_array_311, full_int_array_312, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_153 = paddle._C_ops.gather(transpose_5, slice_158, full_159)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_313 = [154]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_314 = [155]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_159 = paddle._C_ops.slice(parameter_26, [0], full_int_array_313, full_int_array_314, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_154 = paddle._C_ops.gather(transpose_5, slice_159, full_160)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_315 = [155]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_316 = [156]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(parameter_26, [0], full_int_array_315, full_int_array_316, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_155 = paddle._C_ops.gather(transpose_5, slice_160, full_161)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_317 = [156]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_318 = [157]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(parameter_26, [0], full_int_array_317, full_int_array_318, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_156 = paddle._C_ops.gather(transpose_5, slice_161, full_162)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_319 = [157]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_320 = [158]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(parameter_26, [0], full_int_array_319, full_int_array_320, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_157 = paddle._C_ops.gather(transpose_5, slice_162, full_163)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_321 = [158]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_322 = [159]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(parameter_26, [0], full_int_array_321, full_int_array_322, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_158 = paddle._C_ops.gather(transpose_5, slice_163, full_164)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_323 = [159]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_324 = [160]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_164 = paddle._C_ops.slice(parameter_26, [0], full_int_array_323, full_int_array_324, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_159 = paddle._C_ops.gather(transpose_5, slice_164, full_165)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_325 = [160]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_326 = [161]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_165 = paddle._C_ops.slice(parameter_26, [0], full_int_array_325, full_int_array_326, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_160 = paddle._C_ops.gather(transpose_5, slice_165, full_166)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_327 = [161]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_328 = [162]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_166 = paddle._C_ops.slice(parameter_26, [0], full_int_array_327, full_int_array_328, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_161 = paddle._C_ops.gather(transpose_5, slice_166, full_167)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_329 = [162]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_330 = [163]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(parameter_26, [0], full_int_array_329, full_int_array_330, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_162 = paddle._C_ops.gather(transpose_5, slice_167, full_168)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_331 = [163]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_332 = [164]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(parameter_26, [0], full_int_array_331, full_int_array_332, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_163 = paddle._C_ops.gather(transpose_5, slice_168, full_169)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_333 = [164]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_334 = [165]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(parameter_26, [0], full_int_array_333, full_int_array_334, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_164 = paddle._C_ops.gather(transpose_5, slice_169, full_170)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_335 = [165]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_336 = [166]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(parameter_26, [0], full_int_array_335, full_int_array_336, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_165 = paddle._C_ops.gather(transpose_5, slice_170, full_171)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_337 = [166]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_338 = [167]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(parameter_26, [0], full_int_array_337, full_int_array_338, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_166 = paddle._C_ops.gather(transpose_5, slice_171, full_172)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_339 = [167]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_340 = [168]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_172 = paddle._C_ops.slice(parameter_26, [0], full_int_array_339, full_int_array_340, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_167 = paddle._C_ops.gather(transpose_5, slice_172, full_173)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_341 = [168]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_342 = [169]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_173 = paddle._C_ops.slice(parameter_26, [0], full_int_array_341, full_int_array_342, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_168 = paddle._C_ops.gather(transpose_5, slice_173, full_174)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_343 = [169]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_344 = [170]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(parameter_26, [0], full_int_array_343, full_int_array_344, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_169 = paddle._C_ops.gather(transpose_5, slice_174, full_175)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_345 = [170]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_346 = [171]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(parameter_26, [0], full_int_array_345, full_int_array_346, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_170 = paddle._C_ops.gather(transpose_5, slice_175, full_176)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_347 = [171]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_348 = [172]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(parameter_26, [0], full_int_array_347, full_int_array_348, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_171 = paddle._C_ops.gather(transpose_5, slice_176, full_177)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_349 = [172]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_350 = [173]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(parameter_26, [0], full_int_array_349, full_int_array_350, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_172 = paddle._C_ops.gather(transpose_5, slice_177, full_178)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_351 = [173]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_352 = [174]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(parameter_26, [0], full_int_array_351, full_int_array_352, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_173 = paddle._C_ops.gather(transpose_5, slice_178, full_179)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_353 = [174]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_354 = [175]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_179 = paddle._C_ops.slice(parameter_26, [0], full_int_array_353, full_int_array_354, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_174 = paddle._C_ops.gather(transpose_5, slice_179, full_180)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_355 = [175]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_356 = [176]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_180 = paddle._C_ops.slice(parameter_26, [0], full_int_array_355, full_int_array_356, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_175 = paddle._C_ops.gather(transpose_5, slice_180, full_181)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_357 = [176]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_358 = [177]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_181 = paddle._C_ops.slice(parameter_26, [0], full_int_array_357, full_int_array_358, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_182 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_176 = paddle._C_ops.gather(transpose_5, slice_181, full_182)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_359 = [177]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_360 = [178]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_182 = paddle._C_ops.slice(parameter_26, [0], full_int_array_359, full_int_array_360, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_177 = paddle._C_ops.gather(transpose_5, slice_182, full_183)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_361 = [178]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_362 = [179]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_183 = paddle._C_ops.slice(parameter_26, [0], full_int_array_361, full_int_array_362, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_178 = paddle._C_ops.gather(transpose_5, slice_183, full_184)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_363 = [179]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_364 = [180]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_184 = paddle._C_ops.slice(parameter_26, [0], full_int_array_363, full_int_array_364, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_179 = paddle._C_ops.gather(transpose_5, slice_184, full_185)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_365 = [180]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_366 = [181]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_185 = paddle._C_ops.slice(parameter_26, [0], full_int_array_365, full_int_array_366, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_180 = paddle._C_ops.gather(transpose_5, slice_185, full_186)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_367 = [181]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_368 = [182]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_186 = paddle._C_ops.slice(parameter_26, [0], full_int_array_367, full_int_array_368, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_181 = paddle._C_ops.gather(transpose_5, slice_186, full_187)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_369 = [182]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_370 = [183]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_187 = paddle._C_ops.slice(parameter_26, [0], full_int_array_369, full_int_array_370, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_182 = paddle._C_ops.gather(transpose_5, slice_187, full_188)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_371 = [183]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_372 = [184]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_188 = paddle._C_ops.slice(parameter_26, [0], full_int_array_371, full_int_array_372, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_183 = paddle._C_ops.gather(transpose_5, slice_188, full_189)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_373 = [184]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_374 = [185]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_189 = paddle._C_ops.slice(parameter_26, [0], full_int_array_373, full_int_array_374, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_184 = paddle._C_ops.gather(transpose_5, slice_189, full_190)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_375 = [185]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_376 = [186]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_190 = paddle._C_ops.slice(parameter_26, [0], full_int_array_375, full_int_array_376, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_185 = paddle._C_ops.gather(transpose_5, slice_190, full_191)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_377 = [186]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_378 = [187]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_191 = paddle._C_ops.slice(parameter_26, [0], full_int_array_377, full_int_array_378, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_192 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_186 = paddle._C_ops.gather(transpose_5, slice_191, full_192)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_379 = [187]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_380 = [188]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_192 = paddle._C_ops.slice(parameter_26, [0], full_int_array_379, full_int_array_380, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_187 = paddle._C_ops.gather(transpose_5, slice_192, full_193)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_381 = [188]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_382 = [189]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_193 = paddle._C_ops.slice(parameter_26, [0], full_int_array_381, full_int_array_382, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_188 = paddle._C_ops.gather(transpose_5, slice_193, full_194)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_383 = [189]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_384 = [190]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_194 = paddle._C_ops.slice(parameter_26, [0], full_int_array_383, full_int_array_384, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_189 = paddle._C_ops.gather(transpose_5, slice_194, full_195)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_385 = [190]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_386 = [191]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_195 = paddle._C_ops.slice(parameter_26, [0], full_int_array_385, full_int_array_386, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_190 = paddle._C_ops.gather(transpose_5, slice_195, full_196)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_387 = [191]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_388 = [192]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_196 = paddle._C_ops.slice(parameter_26, [0], full_int_array_387, full_int_array_388, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_191 = paddle._C_ops.gather(transpose_5, slice_196, full_197)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_389 = [192]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_390 = [193]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_197 = paddle._C_ops.slice(parameter_26, [0], full_int_array_389, full_int_array_390, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_198 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_192 = paddle._C_ops.gather(transpose_5, slice_197, full_198)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_391 = [193]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_392 = [194]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_198 = paddle._C_ops.slice(parameter_26, [0], full_int_array_391, full_int_array_392, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_199 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_193 = paddle._C_ops.gather(transpose_5, slice_198, full_199)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_393 = [194]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_394 = [195]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_199 = paddle._C_ops.slice(parameter_26, [0], full_int_array_393, full_int_array_394, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_200 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_194 = paddle._C_ops.gather(transpose_5, slice_199, full_200)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_395 = [195]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_396 = [196]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_200 = paddle._C_ops.slice(parameter_26, [0], full_int_array_395, full_int_array_396, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_195 = paddle._C_ops.gather(transpose_5, slice_200, full_201)

        # builtin.combine: ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32]) <- (196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32)
        combine_2 = [gather_0, gather_1, gather_2, gather_3, gather_4, gather_5, gather_6, gather_7, gather_8, gather_9, gather_10, gather_11, gather_12, gather_13, gather_14, gather_15, gather_16, gather_17, gather_18, gather_19, gather_20, gather_21, gather_22, gather_23, gather_24, gather_25, gather_26, gather_27, gather_28, gather_29, gather_30, gather_31, gather_32, gather_33, gather_34, gather_35, gather_36, gather_37, gather_38, gather_39, gather_40, gather_41, gather_42, gather_43, gather_44, gather_45, gather_46, gather_47, gather_48, gather_49, gather_50, gather_51, gather_52, gather_53, gather_54, gather_55, gather_56, gather_57, gather_58, gather_59, gather_60, gather_61, gather_62, gather_63, gather_64, gather_65, gather_66, gather_67, gather_68, gather_69, gather_70, gather_71, gather_72, gather_73, gather_74, gather_75, gather_76, gather_77, gather_78, gather_79, gather_80, gather_81, gather_82, gather_83, gather_84, gather_85, gather_86, gather_87, gather_88, gather_89, gather_90, gather_91, gather_92, gather_93, gather_94, gather_95, gather_96, gather_97, gather_98, gather_99, gather_100, gather_101, gather_102, gather_103, gather_104, gather_105, gather_106, gather_107, gather_108, gather_109, gather_110, gather_111, gather_112, gather_113, gather_114, gather_115, gather_116, gather_117, gather_118, gather_119, gather_120, gather_121, gather_122, gather_123, gather_124, gather_125, gather_126, gather_127, gather_128, gather_129, gather_130, gather_131, gather_132, gather_133, gather_134, gather_135, gather_136, gather_137, gather_138, gather_139, gather_140, gather_141, gather_142, gather_143, gather_144, gather_145, gather_146, gather_147, gather_148, gather_149, gather_150, gather_151, gather_152, gather_153, gather_154, gather_155, gather_156, gather_157, gather_158, gather_159, gather_160, gather_161, gather_162, gather_163, gather_164, gather_165, gather_166, gather_167, gather_168, gather_169, gather_170, gather_171, gather_172, gather_173, gather_174, gather_175, gather_176, gather_177, gather_178, gather_179, gather_180, gather_181, gather_182, gather_183, gather_184, gather_185, gather_186, gather_187, gather_188, gather_189, gather_190, gather_191, gather_192, gather_193, gather_194, gather_195]

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (38416x4xf32) <- ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_202)

        # pd_op.transpose: (4x38416xf32) <- (38416x4xf32)
        transpose_6 = paddle._C_ops.transpose(concat_0, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_397 = [0, 196, 196]

        # pd_op.reshape_: (4x196x196xf32, 0x4x38416xf32) <- (4x38416xf32, 3xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, full_int_array_397), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4x196x196xf32) <- (-1x4x196x16xf32, -1x4x16x196xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_1, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_203 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_1, full_203, float('0'), True)

        # pd_op.add_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 4x196x196xf32)
        add__0 = paddle._C_ops.add_(scale__0, reshape__4)

        # pd_op.softmax_: (-1x4x196x196xf32) <- (-1x4x196x196xf32)
        softmax__0 = paddle._C_ops.softmax_(add__0, -1)

        # pd_op.matmul: (-1x4x196x32xf32) <- (-1x4x196x196xf32, -1x4x196x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax__0, transpose_3, False, False)

        # pd_op.transpose: (-1x196x4x32xf32) <- (-1x4x196x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_0, full_204, full_205]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x196x4x32xf32) <- (-1x196x4x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x128xf32) <- (-1x196x128xf32)
        hardswish_3 = paddle._C_ops.hardswish(reshape__6)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x128xf32, 128x128xf32)
        matmul_3 = paddle._C_ops.matmul(hardswish_3, parameter_27, False, False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_2, flatten_3 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_3, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_2, parameter_28, parameter_29, parameter_30, parameter_31, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_2 = paddle._C_ops.shape(matmul_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_398 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_399 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_201 = paddle._C_ops.slice(shape_2, [0], full_int_array_398, full_int_array_399, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_207 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_201, full_206, full_207]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__30, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__1 = paddle._C_ops.add_(transpose_0, reshape__8)

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_4 = paddle._C_ops.matmul(add__1, parameter_32, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_4, flatten_5 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_4, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_4, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_3 = paddle._C_ops.shape(matmul_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_400 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_401 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_202 = paddle._C_ops.slice(shape_3, [0], full_int_array_400, full_int_array_401, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_208 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_5 = [slice_202, full_208, full_209]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__36, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x256xf32) <- (-1x196x256xf32)
        hardswish_4 = paddle._C_ops.hardswish(reshape__10)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x256xf32, 256x128xf32)
        matmul_5 = paddle._C_ops.matmul(hardswish_4, parameter_37, False, False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_6, flatten_7 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_5, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_6, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_4 = paddle._C_ops.shape(matmul_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_402 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_403 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_203 = paddle._C_ops.slice(shape_4, [0], full_int_array_402, full_int_array_403, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_6 = [slice_203, full_210, full_211]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__42, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__2 = paddle._C_ops.add_(add__1, reshape__12)

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_5 = paddle._C_ops.shape(add__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_404 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_405 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_204 = paddle._C_ops.slice(shape_5, [0], full_int_array_404, full_int_array_405, [1], [0])

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_6 = paddle._C_ops.matmul(add__2, parameter_42, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_8, flatten_9 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_6, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_8, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_6 = paddle._C_ops.shape(matmul_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_406 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_407 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_205 = paddle._C_ops.slice(shape_6, [0], full_int_array_406, full_int_array_407, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_7 = [slice_205, full_212, full_213]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__48, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_204, full_214, full_215, full_216]

        # pd_op.reshape_: (-1x196x4x64xf32, 0x-1x196x256xf32) <- (-1x196x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__14, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_408 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32]) <- (-1x196x4x64xf32, 3xi64, 1xi32)
        split_1 = paddle._C_ops.split(reshape__16, full_int_array_408, full_217)

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_206 = split_1[0]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_8 = paddle._C_ops.transpose(slice_206, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x16xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_207 = split_1[1]

        # pd_op.transpose: (-1x4x196x16xf32) <- (-1x196x4x16xf32)
        transpose_9 = paddle._C_ops.transpose(slice_207, [0, 2, 1, 3])

        # builtin.slice: (-1x196x4x32xf32) <- ([-1x196x4x16xf32, -1x196x4x16xf32, -1x196x4x32xf32])
        slice_208 = split_1[2]

        # pd_op.transpose: (-1x4x196x32xf32) <- (-1x196x4x32xf32)
        transpose_10 = paddle._C_ops.transpose(slice_208, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x16x196xf32) <- (-1x4x196x16xf32)
        transpose_11 = paddle._C_ops.transpose(transpose_9, [0, 1, 3, 2])

        # pd_op.transpose: (196x4xf32) <- (4x196xf32)
        transpose_12 = paddle._C_ops.transpose(parameter_47, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_409 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_410 = [1]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_209 = paddle._C_ops.slice(parameter_48, [0], full_int_array_409, full_int_array_410, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_196 = paddle._C_ops.gather(transpose_12, slice_209, full_218)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_411 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_412 = [2]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_210 = paddle._C_ops.slice(parameter_48, [0], full_int_array_411, full_int_array_412, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_197 = paddle._C_ops.gather(transpose_12, slice_210, full_219)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_413 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_414 = [3]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_211 = paddle._C_ops.slice(parameter_48, [0], full_int_array_413, full_int_array_414, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_198 = paddle._C_ops.gather(transpose_12, slice_211, full_220)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_415 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_416 = [4]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_212 = paddle._C_ops.slice(parameter_48, [0], full_int_array_415, full_int_array_416, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_221 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_199 = paddle._C_ops.gather(transpose_12, slice_212, full_221)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_417 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_418 = [5]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_213 = paddle._C_ops.slice(parameter_48, [0], full_int_array_417, full_int_array_418, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_222 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_200 = paddle._C_ops.gather(transpose_12, slice_213, full_222)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_419 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_420 = [6]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_214 = paddle._C_ops.slice(parameter_48, [0], full_int_array_419, full_int_array_420, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_201 = paddle._C_ops.gather(transpose_12, slice_214, full_223)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_421 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_422 = [7]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_215 = paddle._C_ops.slice(parameter_48, [0], full_int_array_421, full_int_array_422, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_202 = paddle._C_ops.gather(transpose_12, slice_215, full_224)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_423 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_424 = [8]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_216 = paddle._C_ops.slice(parameter_48, [0], full_int_array_423, full_int_array_424, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_203 = paddle._C_ops.gather(transpose_12, slice_216, full_225)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_425 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_426 = [9]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_217 = paddle._C_ops.slice(parameter_48, [0], full_int_array_425, full_int_array_426, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_204 = paddle._C_ops.gather(transpose_12, slice_217, full_226)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_427 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_428 = [10]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_218 = paddle._C_ops.slice(parameter_48, [0], full_int_array_427, full_int_array_428, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_205 = paddle._C_ops.gather(transpose_12, slice_218, full_227)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_429 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_430 = [11]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_219 = paddle._C_ops.slice(parameter_48, [0], full_int_array_429, full_int_array_430, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_206 = paddle._C_ops.gather(transpose_12, slice_219, full_228)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_431 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_432 = [12]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_220 = paddle._C_ops.slice(parameter_48, [0], full_int_array_431, full_int_array_432, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_207 = paddle._C_ops.gather(transpose_12, slice_220, full_229)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_433 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_434 = [13]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_221 = paddle._C_ops.slice(parameter_48, [0], full_int_array_433, full_int_array_434, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_230 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_208 = paddle._C_ops.gather(transpose_12, slice_221, full_230)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_435 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_436 = [14]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_222 = paddle._C_ops.slice(parameter_48, [0], full_int_array_435, full_int_array_436, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_209 = paddle._C_ops.gather(transpose_12, slice_222, full_231)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_437 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_438 = [15]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_223 = paddle._C_ops.slice(parameter_48, [0], full_int_array_437, full_int_array_438, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_210 = paddle._C_ops.gather(transpose_12, slice_223, full_232)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_439 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_440 = [16]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_224 = paddle._C_ops.slice(parameter_48, [0], full_int_array_439, full_int_array_440, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_211 = paddle._C_ops.gather(transpose_12, slice_224, full_233)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_441 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_442 = [17]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_225 = paddle._C_ops.slice(parameter_48, [0], full_int_array_441, full_int_array_442, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_212 = paddle._C_ops.gather(transpose_12, slice_225, full_234)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_443 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_444 = [18]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_226 = paddle._C_ops.slice(parameter_48, [0], full_int_array_443, full_int_array_444, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_213 = paddle._C_ops.gather(transpose_12, slice_226, full_235)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_445 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_446 = [19]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_227 = paddle._C_ops.slice(parameter_48, [0], full_int_array_445, full_int_array_446, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_214 = paddle._C_ops.gather(transpose_12, slice_227, full_236)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_447 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_448 = [20]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_228 = paddle._C_ops.slice(parameter_48, [0], full_int_array_447, full_int_array_448, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_237 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_215 = paddle._C_ops.gather(transpose_12, slice_228, full_237)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_449 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_450 = [21]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_229 = paddle._C_ops.slice(parameter_48, [0], full_int_array_449, full_int_array_450, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_216 = paddle._C_ops.gather(transpose_12, slice_229, full_238)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_451 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_452 = [22]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_230 = paddle._C_ops.slice(parameter_48, [0], full_int_array_451, full_int_array_452, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_217 = paddle._C_ops.gather(transpose_12, slice_230, full_239)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_453 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_454 = [23]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_231 = paddle._C_ops.slice(parameter_48, [0], full_int_array_453, full_int_array_454, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_218 = paddle._C_ops.gather(transpose_12, slice_231, full_240)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_455 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_456 = [24]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_232 = paddle._C_ops.slice(parameter_48, [0], full_int_array_455, full_int_array_456, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_219 = paddle._C_ops.gather(transpose_12, slice_232, full_241)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_457 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_458 = [25]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_233 = paddle._C_ops.slice(parameter_48, [0], full_int_array_457, full_int_array_458, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_242 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_220 = paddle._C_ops.gather(transpose_12, slice_233, full_242)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_459 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_460 = [26]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_234 = paddle._C_ops.slice(parameter_48, [0], full_int_array_459, full_int_array_460, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_221 = paddle._C_ops.gather(transpose_12, slice_234, full_243)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_461 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_462 = [27]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_235 = paddle._C_ops.slice(parameter_48, [0], full_int_array_461, full_int_array_462, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_222 = paddle._C_ops.gather(transpose_12, slice_235, full_244)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_463 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_464 = [28]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_236 = paddle._C_ops.slice(parameter_48, [0], full_int_array_463, full_int_array_464, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_223 = paddle._C_ops.gather(transpose_12, slice_236, full_245)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_465 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_466 = [29]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_237 = paddle._C_ops.slice(parameter_48, [0], full_int_array_465, full_int_array_466, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_224 = paddle._C_ops.gather(transpose_12, slice_237, full_246)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_467 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_468 = [30]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_238 = paddle._C_ops.slice(parameter_48, [0], full_int_array_467, full_int_array_468, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_225 = paddle._C_ops.gather(transpose_12, slice_238, full_247)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_469 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_470 = [31]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_239 = paddle._C_ops.slice(parameter_48, [0], full_int_array_469, full_int_array_470, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_226 = paddle._C_ops.gather(transpose_12, slice_239, full_248)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_471 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_472 = [32]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_240 = paddle._C_ops.slice(parameter_48, [0], full_int_array_471, full_int_array_472, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_227 = paddle._C_ops.gather(transpose_12, slice_240, full_249)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_473 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_474 = [33]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_241 = paddle._C_ops.slice(parameter_48, [0], full_int_array_473, full_int_array_474, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_228 = paddle._C_ops.gather(transpose_12, slice_241, full_250)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_475 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_476 = [34]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_242 = paddle._C_ops.slice(parameter_48, [0], full_int_array_475, full_int_array_476, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_229 = paddle._C_ops.gather(transpose_12, slice_242, full_251)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_477 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_478 = [35]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_243 = paddle._C_ops.slice(parameter_48, [0], full_int_array_477, full_int_array_478, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_252 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_230 = paddle._C_ops.gather(transpose_12, slice_243, full_252)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_479 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_480 = [36]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_244 = paddle._C_ops.slice(parameter_48, [0], full_int_array_479, full_int_array_480, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_231 = paddle._C_ops.gather(transpose_12, slice_244, full_253)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_481 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_482 = [37]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_245 = paddle._C_ops.slice(parameter_48, [0], full_int_array_481, full_int_array_482, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_232 = paddle._C_ops.gather(transpose_12, slice_245, full_254)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_483 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_484 = [38]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_246 = paddle._C_ops.slice(parameter_48, [0], full_int_array_483, full_int_array_484, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_233 = paddle._C_ops.gather(transpose_12, slice_246, full_255)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_485 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_486 = [39]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_247 = paddle._C_ops.slice(parameter_48, [0], full_int_array_485, full_int_array_486, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_234 = paddle._C_ops.gather(transpose_12, slice_247, full_256)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_487 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_488 = [40]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_248 = paddle._C_ops.slice(parameter_48, [0], full_int_array_487, full_int_array_488, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_235 = paddle._C_ops.gather(transpose_12, slice_248, full_257)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_489 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_490 = [41]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_249 = paddle._C_ops.slice(parameter_48, [0], full_int_array_489, full_int_array_490, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_236 = paddle._C_ops.gather(transpose_12, slice_249, full_258)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_491 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_492 = [42]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_250 = paddle._C_ops.slice(parameter_48, [0], full_int_array_491, full_int_array_492, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_237 = paddle._C_ops.gather(transpose_12, slice_250, full_259)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_493 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_494 = [43]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_251 = paddle._C_ops.slice(parameter_48, [0], full_int_array_493, full_int_array_494, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_260 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_238 = paddle._C_ops.gather(transpose_12, slice_251, full_260)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_495 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_496 = [44]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_252 = paddle._C_ops.slice(parameter_48, [0], full_int_array_495, full_int_array_496, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_239 = paddle._C_ops.gather(transpose_12, slice_252, full_261)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_497 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_498 = [45]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_253 = paddle._C_ops.slice(parameter_48, [0], full_int_array_497, full_int_array_498, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_240 = paddle._C_ops.gather(transpose_12, slice_253, full_262)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_499 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_500 = [46]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_254 = paddle._C_ops.slice(parameter_48, [0], full_int_array_499, full_int_array_500, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_241 = paddle._C_ops.gather(transpose_12, slice_254, full_263)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_501 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_502 = [47]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_255 = paddle._C_ops.slice(parameter_48, [0], full_int_array_501, full_int_array_502, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_264 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_242 = paddle._C_ops.gather(transpose_12, slice_255, full_264)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_503 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_504 = [48]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_256 = paddle._C_ops.slice(parameter_48, [0], full_int_array_503, full_int_array_504, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_243 = paddle._C_ops.gather(transpose_12, slice_256, full_265)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_505 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_506 = [49]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_257 = paddle._C_ops.slice(parameter_48, [0], full_int_array_505, full_int_array_506, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_244 = paddle._C_ops.gather(transpose_12, slice_257, full_266)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_507 = [49]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_508 = [50]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_258 = paddle._C_ops.slice(parameter_48, [0], full_int_array_507, full_int_array_508, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_267 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_245 = paddle._C_ops.gather(transpose_12, slice_258, full_267)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_509 = [50]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_510 = [51]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_259 = paddle._C_ops.slice(parameter_48, [0], full_int_array_509, full_int_array_510, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_268 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_246 = paddle._C_ops.gather(transpose_12, slice_259, full_268)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_511 = [51]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_512 = [52]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_260 = paddle._C_ops.slice(parameter_48, [0], full_int_array_511, full_int_array_512, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_269 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_247 = paddle._C_ops.gather(transpose_12, slice_260, full_269)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_513 = [52]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_514 = [53]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_261 = paddle._C_ops.slice(parameter_48, [0], full_int_array_513, full_int_array_514, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_270 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_248 = paddle._C_ops.gather(transpose_12, slice_261, full_270)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_515 = [53]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_516 = [54]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_262 = paddle._C_ops.slice(parameter_48, [0], full_int_array_515, full_int_array_516, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_271 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_249 = paddle._C_ops.gather(transpose_12, slice_262, full_271)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_517 = [54]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_518 = [55]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_263 = paddle._C_ops.slice(parameter_48, [0], full_int_array_517, full_int_array_518, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_272 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_250 = paddle._C_ops.gather(transpose_12, slice_263, full_272)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_519 = [55]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_520 = [56]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_264 = paddle._C_ops.slice(parameter_48, [0], full_int_array_519, full_int_array_520, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_273 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_251 = paddle._C_ops.gather(transpose_12, slice_264, full_273)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_521 = [56]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_522 = [57]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_265 = paddle._C_ops.slice(parameter_48, [0], full_int_array_521, full_int_array_522, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_274 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_252 = paddle._C_ops.gather(transpose_12, slice_265, full_274)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_523 = [57]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_524 = [58]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_266 = paddle._C_ops.slice(parameter_48, [0], full_int_array_523, full_int_array_524, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_275 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_253 = paddle._C_ops.gather(transpose_12, slice_266, full_275)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_525 = [58]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_526 = [59]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_267 = paddle._C_ops.slice(parameter_48, [0], full_int_array_525, full_int_array_526, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_276 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_254 = paddle._C_ops.gather(transpose_12, slice_267, full_276)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_527 = [59]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_528 = [60]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_268 = paddle._C_ops.slice(parameter_48, [0], full_int_array_527, full_int_array_528, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_277 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_255 = paddle._C_ops.gather(transpose_12, slice_268, full_277)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_529 = [60]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_530 = [61]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_269 = paddle._C_ops.slice(parameter_48, [0], full_int_array_529, full_int_array_530, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_278 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_256 = paddle._C_ops.gather(transpose_12, slice_269, full_278)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_531 = [61]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_532 = [62]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_270 = paddle._C_ops.slice(parameter_48, [0], full_int_array_531, full_int_array_532, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_279 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_257 = paddle._C_ops.gather(transpose_12, slice_270, full_279)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_533 = [62]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_534 = [63]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_271 = paddle._C_ops.slice(parameter_48, [0], full_int_array_533, full_int_array_534, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_280 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_258 = paddle._C_ops.gather(transpose_12, slice_271, full_280)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_535 = [63]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_536 = [64]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_272 = paddle._C_ops.slice(parameter_48, [0], full_int_array_535, full_int_array_536, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_281 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_259 = paddle._C_ops.gather(transpose_12, slice_272, full_281)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_537 = [64]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_538 = [65]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_273 = paddle._C_ops.slice(parameter_48, [0], full_int_array_537, full_int_array_538, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_282 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_260 = paddle._C_ops.gather(transpose_12, slice_273, full_282)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_539 = [65]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_540 = [66]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_274 = paddle._C_ops.slice(parameter_48, [0], full_int_array_539, full_int_array_540, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_283 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_261 = paddle._C_ops.gather(transpose_12, slice_274, full_283)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_541 = [66]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_542 = [67]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_275 = paddle._C_ops.slice(parameter_48, [0], full_int_array_541, full_int_array_542, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_284 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_262 = paddle._C_ops.gather(transpose_12, slice_275, full_284)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_543 = [67]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_544 = [68]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_276 = paddle._C_ops.slice(parameter_48, [0], full_int_array_543, full_int_array_544, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_285 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_263 = paddle._C_ops.gather(transpose_12, slice_276, full_285)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_545 = [68]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_546 = [69]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_277 = paddle._C_ops.slice(parameter_48, [0], full_int_array_545, full_int_array_546, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_286 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_264 = paddle._C_ops.gather(transpose_12, slice_277, full_286)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_547 = [69]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_548 = [70]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_278 = paddle._C_ops.slice(parameter_48, [0], full_int_array_547, full_int_array_548, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_287 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_265 = paddle._C_ops.gather(transpose_12, slice_278, full_287)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_549 = [70]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_550 = [71]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_279 = paddle._C_ops.slice(parameter_48, [0], full_int_array_549, full_int_array_550, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_288 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_266 = paddle._C_ops.gather(transpose_12, slice_279, full_288)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_551 = [71]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_552 = [72]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_280 = paddle._C_ops.slice(parameter_48, [0], full_int_array_551, full_int_array_552, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_289 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_267 = paddle._C_ops.gather(transpose_12, slice_280, full_289)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_553 = [72]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_554 = [73]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_281 = paddle._C_ops.slice(parameter_48, [0], full_int_array_553, full_int_array_554, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_290 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_268 = paddle._C_ops.gather(transpose_12, slice_281, full_290)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_555 = [73]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_556 = [74]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_282 = paddle._C_ops.slice(parameter_48, [0], full_int_array_555, full_int_array_556, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_291 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_269 = paddle._C_ops.gather(transpose_12, slice_282, full_291)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_557 = [74]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_558 = [75]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_283 = paddle._C_ops.slice(parameter_48, [0], full_int_array_557, full_int_array_558, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_292 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_270 = paddle._C_ops.gather(transpose_12, slice_283, full_292)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_559 = [75]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_560 = [76]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_284 = paddle._C_ops.slice(parameter_48, [0], full_int_array_559, full_int_array_560, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_293 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_271 = paddle._C_ops.gather(transpose_12, slice_284, full_293)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_561 = [76]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_562 = [77]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_285 = paddle._C_ops.slice(parameter_48, [0], full_int_array_561, full_int_array_562, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_294 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_272 = paddle._C_ops.gather(transpose_12, slice_285, full_294)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_563 = [77]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_564 = [78]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_286 = paddle._C_ops.slice(parameter_48, [0], full_int_array_563, full_int_array_564, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_295 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_273 = paddle._C_ops.gather(transpose_12, slice_286, full_295)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_565 = [78]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_566 = [79]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_287 = paddle._C_ops.slice(parameter_48, [0], full_int_array_565, full_int_array_566, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_296 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_274 = paddle._C_ops.gather(transpose_12, slice_287, full_296)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_567 = [79]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_568 = [80]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_288 = paddle._C_ops.slice(parameter_48, [0], full_int_array_567, full_int_array_568, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_297 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_275 = paddle._C_ops.gather(transpose_12, slice_288, full_297)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_569 = [80]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_570 = [81]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_289 = paddle._C_ops.slice(parameter_48, [0], full_int_array_569, full_int_array_570, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_298 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_276 = paddle._C_ops.gather(transpose_12, slice_289, full_298)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_571 = [81]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_572 = [82]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_290 = paddle._C_ops.slice(parameter_48, [0], full_int_array_571, full_int_array_572, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_299 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_277 = paddle._C_ops.gather(transpose_12, slice_290, full_299)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_573 = [82]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_574 = [83]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_291 = paddle._C_ops.slice(parameter_48, [0], full_int_array_573, full_int_array_574, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_300 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_278 = paddle._C_ops.gather(transpose_12, slice_291, full_300)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_575 = [83]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_576 = [84]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_292 = paddle._C_ops.slice(parameter_48, [0], full_int_array_575, full_int_array_576, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_301 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_279 = paddle._C_ops.gather(transpose_12, slice_292, full_301)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_577 = [84]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_578 = [85]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_293 = paddle._C_ops.slice(parameter_48, [0], full_int_array_577, full_int_array_578, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_302 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_280 = paddle._C_ops.gather(transpose_12, slice_293, full_302)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_579 = [85]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_580 = [86]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_294 = paddle._C_ops.slice(parameter_48, [0], full_int_array_579, full_int_array_580, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_303 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_281 = paddle._C_ops.gather(transpose_12, slice_294, full_303)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_581 = [86]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_582 = [87]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_295 = paddle._C_ops.slice(parameter_48, [0], full_int_array_581, full_int_array_582, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_304 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_282 = paddle._C_ops.gather(transpose_12, slice_295, full_304)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_583 = [87]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_584 = [88]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_296 = paddle._C_ops.slice(parameter_48, [0], full_int_array_583, full_int_array_584, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_305 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_283 = paddle._C_ops.gather(transpose_12, slice_296, full_305)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_585 = [88]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_586 = [89]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_297 = paddle._C_ops.slice(parameter_48, [0], full_int_array_585, full_int_array_586, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_306 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_284 = paddle._C_ops.gather(transpose_12, slice_297, full_306)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_587 = [89]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_588 = [90]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_298 = paddle._C_ops.slice(parameter_48, [0], full_int_array_587, full_int_array_588, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_307 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_285 = paddle._C_ops.gather(transpose_12, slice_298, full_307)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_589 = [90]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_590 = [91]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_299 = paddle._C_ops.slice(parameter_48, [0], full_int_array_589, full_int_array_590, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_308 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_286 = paddle._C_ops.gather(transpose_12, slice_299, full_308)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_591 = [91]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_592 = [92]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_300 = paddle._C_ops.slice(parameter_48, [0], full_int_array_591, full_int_array_592, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_309 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_287 = paddle._C_ops.gather(transpose_12, slice_300, full_309)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_593 = [92]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_594 = [93]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_301 = paddle._C_ops.slice(parameter_48, [0], full_int_array_593, full_int_array_594, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_310 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_288 = paddle._C_ops.gather(transpose_12, slice_301, full_310)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_595 = [93]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_596 = [94]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_302 = paddle._C_ops.slice(parameter_48, [0], full_int_array_595, full_int_array_596, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_311 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_289 = paddle._C_ops.gather(transpose_12, slice_302, full_311)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_597 = [94]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_598 = [95]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_303 = paddle._C_ops.slice(parameter_48, [0], full_int_array_597, full_int_array_598, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_312 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_290 = paddle._C_ops.gather(transpose_12, slice_303, full_312)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_599 = [95]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_600 = [96]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_304 = paddle._C_ops.slice(parameter_48, [0], full_int_array_599, full_int_array_600, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_313 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_291 = paddle._C_ops.gather(transpose_12, slice_304, full_313)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_601 = [96]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_602 = [97]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_305 = paddle._C_ops.slice(parameter_48, [0], full_int_array_601, full_int_array_602, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_314 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_292 = paddle._C_ops.gather(transpose_12, slice_305, full_314)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_603 = [97]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_604 = [98]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_306 = paddle._C_ops.slice(parameter_48, [0], full_int_array_603, full_int_array_604, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_315 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_293 = paddle._C_ops.gather(transpose_12, slice_306, full_315)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_605 = [98]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_606 = [99]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_307 = paddle._C_ops.slice(parameter_48, [0], full_int_array_605, full_int_array_606, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_316 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_294 = paddle._C_ops.gather(transpose_12, slice_307, full_316)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_607 = [99]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_608 = [100]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_308 = paddle._C_ops.slice(parameter_48, [0], full_int_array_607, full_int_array_608, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_317 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_295 = paddle._C_ops.gather(transpose_12, slice_308, full_317)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_609 = [100]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_610 = [101]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_309 = paddle._C_ops.slice(parameter_48, [0], full_int_array_609, full_int_array_610, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_318 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_296 = paddle._C_ops.gather(transpose_12, slice_309, full_318)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_611 = [101]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_612 = [102]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_310 = paddle._C_ops.slice(parameter_48, [0], full_int_array_611, full_int_array_612, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_319 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_297 = paddle._C_ops.gather(transpose_12, slice_310, full_319)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_613 = [102]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_614 = [103]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_311 = paddle._C_ops.slice(parameter_48, [0], full_int_array_613, full_int_array_614, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_320 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_298 = paddle._C_ops.gather(transpose_12, slice_311, full_320)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_615 = [103]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_616 = [104]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_312 = paddle._C_ops.slice(parameter_48, [0], full_int_array_615, full_int_array_616, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_321 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_299 = paddle._C_ops.gather(transpose_12, slice_312, full_321)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_617 = [104]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_618 = [105]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_313 = paddle._C_ops.slice(parameter_48, [0], full_int_array_617, full_int_array_618, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_322 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_300 = paddle._C_ops.gather(transpose_12, slice_313, full_322)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_619 = [105]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_620 = [106]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_314 = paddle._C_ops.slice(parameter_48, [0], full_int_array_619, full_int_array_620, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_323 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_301 = paddle._C_ops.gather(transpose_12, slice_314, full_323)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_621 = [106]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_622 = [107]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_315 = paddle._C_ops.slice(parameter_48, [0], full_int_array_621, full_int_array_622, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_324 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_302 = paddle._C_ops.gather(transpose_12, slice_315, full_324)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_623 = [107]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_624 = [108]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_316 = paddle._C_ops.slice(parameter_48, [0], full_int_array_623, full_int_array_624, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_325 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_303 = paddle._C_ops.gather(transpose_12, slice_316, full_325)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_625 = [108]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_626 = [109]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_317 = paddle._C_ops.slice(parameter_48, [0], full_int_array_625, full_int_array_626, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_326 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_304 = paddle._C_ops.gather(transpose_12, slice_317, full_326)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_627 = [109]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_628 = [110]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_318 = paddle._C_ops.slice(parameter_48, [0], full_int_array_627, full_int_array_628, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_327 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_305 = paddle._C_ops.gather(transpose_12, slice_318, full_327)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_629 = [110]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_630 = [111]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_319 = paddle._C_ops.slice(parameter_48, [0], full_int_array_629, full_int_array_630, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_328 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_306 = paddle._C_ops.gather(transpose_12, slice_319, full_328)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_631 = [111]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_632 = [112]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_320 = paddle._C_ops.slice(parameter_48, [0], full_int_array_631, full_int_array_632, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_329 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_307 = paddle._C_ops.gather(transpose_12, slice_320, full_329)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_633 = [112]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_634 = [113]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_321 = paddle._C_ops.slice(parameter_48, [0], full_int_array_633, full_int_array_634, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_330 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_308 = paddle._C_ops.gather(transpose_12, slice_321, full_330)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_635 = [113]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_636 = [114]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_322 = paddle._C_ops.slice(parameter_48, [0], full_int_array_635, full_int_array_636, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_331 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_309 = paddle._C_ops.gather(transpose_12, slice_322, full_331)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_637 = [114]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_638 = [115]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_323 = paddle._C_ops.slice(parameter_48, [0], full_int_array_637, full_int_array_638, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_332 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_310 = paddle._C_ops.gather(transpose_12, slice_323, full_332)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_639 = [115]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_640 = [116]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_324 = paddle._C_ops.slice(parameter_48, [0], full_int_array_639, full_int_array_640, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_333 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_311 = paddle._C_ops.gather(transpose_12, slice_324, full_333)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_641 = [116]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_642 = [117]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_325 = paddle._C_ops.slice(parameter_48, [0], full_int_array_641, full_int_array_642, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_334 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_312 = paddle._C_ops.gather(transpose_12, slice_325, full_334)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_643 = [117]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_644 = [118]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_326 = paddle._C_ops.slice(parameter_48, [0], full_int_array_643, full_int_array_644, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_335 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_313 = paddle._C_ops.gather(transpose_12, slice_326, full_335)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_645 = [118]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_646 = [119]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_327 = paddle._C_ops.slice(parameter_48, [0], full_int_array_645, full_int_array_646, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_336 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_314 = paddle._C_ops.gather(transpose_12, slice_327, full_336)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_647 = [119]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_648 = [120]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_328 = paddle._C_ops.slice(parameter_48, [0], full_int_array_647, full_int_array_648, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_337 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_315 = paddle._C_ops.gather(transpose_12, slice_328, full_337)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_649 = [120]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_650 = [121]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_329 = paddle._C_ops.slice(parameter_48, [0], full_int_array_649, full_int_array_650, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_338 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_316 = paddle._C_ops.gather(transpose_12, slice_329, full_338)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_651 = [121]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_652 = [122]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_330 = paddle._C_ops.slice(parameter_48, [0], full_int_array_651, full_int_array_652, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_339 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_317 = paddle._C_ops.gather(transpose_12, slice_330, full_339)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_653 = [122]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_654 = [123]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_331 = paddle._C_ops.slice(parameter_48, [0], full_int_array_653, full_int_array_654, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_340 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_318 = paddle._C_ops.gather(transpose_12, slice_331, full_340)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_655 = [123]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_656 = [124]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_332 = paddle._C_ops.slice(parameter_48, [0], full_int_array_655, full_int_array_656, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_341 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_319 = paddle._C_ops.gather(transpose_12, slice_332, full_341)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_657 = [124]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_658 = [125]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_333 = paddle._C_ops.slice(parameter_48, [0], full_int_array_657, full_int_array_658, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_342 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_320 = paddle._C_ops.gather(transpose_12, slice_333, full_342)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_659 = [125]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_660 = [126]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_334 = paddle._C_ops.slice(parameter_48, [0], full_int_array_659, full_int_array_660, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_343 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_321 = paddle._C_ops.gather(transpose_12, slice_334, full_343)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_661 = [126]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_662 = [127]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_335 = paddle._C_ops.slice(parameter_48, [0], full_int_array_661, full_int_array_662, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_344 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_322 = paddle._C_ops.gather(transpose_12, slice_335, full_344)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_663 = [127]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_664 = [128]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_336 = paddle._C_ops.slice(parameter_48, [0], full_int_array_663, full_int_array_664, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_345 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_323 = paddle._C_ops.gather(transpose_12, slice_336, full_345)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_665 = [128]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_666 = [129]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_337 = paddle._C_ops.slice(parameter_48, [0], full_int_array_665, full_int_array_666, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_346 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_324 = paddle._C_ops.gather(transpose_12, slice_337, full_346)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_667 = [129]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_668 = [130]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_338 = paddle._C_ops.slice(parameter_48, [0], full_int_array_667, full_int_array_668, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_347 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_325 = paddle._C_ops.gather(transpose_12, slice_338, full_347)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_669 = [130]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_670 = [131]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_339 = paddle._C_ops.slice(parameter_48, [0], full_int_array_669, full_int_array_670, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_348 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_326 = paddle._C_ops.gather(transpose_12, slice_339, full_348)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_671 = [131]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_672 = [132]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_340 = paddle._C_ops.slice(parameter_48, [0], full_int_array_671, full_int_array_672, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_349 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_327 = paddle._C_ops.gather(transpose_12, slice_340, full_349)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_673 = [132]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_674 = [133]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_341 = paddle._C_ops.slice(parameter_48, [0], full_int_array_673, full_int_array_674, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_350 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_328 = paddle._C_ops.gather(transpose_12, slice_341, full_350)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_675 = [133]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_676 = [134]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_342 = paddle._C_ops.slice(parameter_48, [0], full_int_array_675, full_int_array_676, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_351 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_329 = paddle._C_ops.gather(transpose_12, slice_342, full_351)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_677 = [134]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_678 = [135]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_343 = paddle._C_ops.slice(parameter_48, [0], full_int_array_677, full_int_array_678, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_352 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_330 = paddle._C_ops.gather(transpose_12, slice_343, full_352)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_679 = [135]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_680 = [136]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_344 = paddle._C_ops.slice(parameter_48, [0], full_int_array_679, full_int_array_680, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_353 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_331 = paddle._C_ops.gather(transpose_12, slice_344, full_353)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_681 = [136]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_682 = [137]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_345 = paddle._C_ops.slice(parameter_48, [0], full_int_array_681, full_int_array_682, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_354 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_332 = paddle._C_ops.gather(transpose_12, slice_345, full_354)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_683 = [137]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_684 = [138]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_346 = paddle._C_ops.slice(parameter_48, [0], full_int_array_683, full_int_array_684, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_355 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_333 = paddle._C_ops.gather(transpose_12, slice_346, full_355)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_685 = [138]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_686 = [139]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_347 = paddle._C_ops.slice(parameter_48, [0], full_int_array_685, full_int_array_686, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_356 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_334 = paddle._C_ops.gather(transpose_12, slice_347, full_356)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_687 = [139]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_688 = [140]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_348 = paddle._C_ops.slice(parameter_48, [0], full_int_array_687, full_int_array_688, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_357 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_335 = paddle._C_ops.gather(transpose_12, slice_348, full_357)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_689 = [140]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_690 = [141]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_349 = paddle._C_ops.slice(parameter_48, [0], full_int_array_689, full_int_array_690, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_358 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_336 = paddle._C_ops.gather(transpose_12, slice_349, full_358)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_691 = [141]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_692 = [142]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_350 = paddle._C_ops.slice(parameter_48, [0], full_int_array_691, full_int_array_692, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_359 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_337 = paddle._C_ops.gather(transpose_12, slice_350, full_359)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_693 = [142]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_694 = [143]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_351 = paddle._C_ops.slice(parameter_48, [0], full_int_array_693, full_int_array_694, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_360 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_338 = paddle._C_ops.gather(transpose_12, slice_351, full_360)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_695 = [143]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_696 = [144]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_352 = paddle._C_ops.slice(parameter_48, [0], full_int_array_695, full_int_array_696, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_361 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_339 = paddle._C_ops.gather(transpose_12, slice_352, full_361)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_697 = [144]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_698 = [145]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_353 = paddle._C_ops.slice(parameter_48, [0], full_int_array_697, full_int_array_698, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_362 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_340 = paddle._C_ops.gather(transpose_12, slice_353, full_362)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_699 = [145]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_700 = [146]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_354 = paddle._C_ops.slice(parameter_48, [0], full_int_array_699, full_int_array_700, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_363 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_341 = paddle._C_ops.gather(transpose_12, slice_354, full_363)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_701 = [146]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_702 = [147]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_355 = paddle._C_ops.slice(parameter_48, [0], full_int_array_701, full_int_array_702, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_364 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_342 = paddle._C_ops.gather(transpose_12, slice_355, full_364)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_703 = [147]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_704 = [148]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_356 = paddle._C_ops.slice(parameter_48, [0], full_int_array_703, full_int_array_704, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_365 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_343 = paddle._C_ops.gather(transpose_12, slice_356, full_365)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_705 = [148]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_706 = [149]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_357 = paddle._C_ops.slice(parameter_48, [0], full_int_array_705, full_int_array_706, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_366 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_344 = paddle._C_ops.gather(transpose_12, slice_357, full_366)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_707 = [149]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_708 = [150]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_358 = paddle._C_ops.slice(parameter_48, [0], full_int_array_707, full_int_array_708, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_367 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_345 = paddle._C_ops.gather(transpose_12, slice_358, full_367)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_709 = [150]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_710 = [151]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_359 = paddle._C_ops.slice(parameter_48, [0], full_int_array_709, full_int_array_710, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_368 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_346 = paddle._C_ops.gather(transpose_12, slice_359, full_368)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_711 = [151]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_712 = [152]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_360 = paddle._C_ops.slice(parameter_48, [0], full_int_array_711, full_int_array_712, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_369 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_347 = paddle._C_ops.gather(transpose_12, slice_360, full_369)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_713 = [152]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_714 = [153]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_361 = paddle._C_ops.slice(parameter_48, [0], full_int_array_713, full_int_array_714, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_370 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_348 = paddle._C_ops.gather(transpose_12, slice_361, full_370)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_715 = [153]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_716 = [154]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_362 = paddle._C_ops.slice(parameter_48, [0], full_int_array_715, full_int_array_716, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_371 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_349 = paddle._C_ops.gather(transpose_12, slice_362, full_371)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_717 = [154]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_718 = [155]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_363 = paddle._C_ops.slice(parameter_48, [0], full_int_array_717, full_int_array_718, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_372 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_350 = paddle._C_ops.gather(transpose_12, slice_363, full_372)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_719 = [155]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_720 = [156]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_364 = paddle._C_ops.slice(parameter_48, [0], full_int_array_719, full_int_array_720, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_373 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_351 = paddle._C_ops.gather(transpose_12, slice_364, full_373)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_721 = [156]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_722 = [157]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_365 = paddle._C_ops.slice(parameter_48, [0], full_int_array_721, full_int_array_722, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_374 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_352 = paddle._C_ops.gather(transpose_12, slice_365, full_374)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_723 = [157]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_724 = [158]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_366 = paddle._C_ops.slice(parameter_48, [0], full_int_array_723, full_int_array_724, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_375 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_353 = paddle._C_ops.gather(transpose_12, slice_366, full_375)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_725 = [158]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_726 = [159]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_367 = paddle._C_ops.slice(parameter_48, [0], full_int_array_725, full_int_array_726, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_376 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_354 = paddle._C_ops.gather(transpose_12, slice_367, full_376)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_727 = [159]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_728 = [160]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_368 = paddle._C_ops.slice(parameter_48, [0], full_int_array_727, full_int_array_728, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_377 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_355 = paddle._C_ops.gather(transpose_12, slice_368, full_377)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_729 = [160]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_730 = [161]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_369 = paddle._C_ops.slice(parameter_48, [0], full_int_array_729, full_int_array_730, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_378 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_356 = paddle._C_ops.gather(transpose_12, slice_369, full_378)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_731 = [161]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_732 = [162]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_370 = paddle._C_ops.slice(parameter_48, [0], full_int_array_731, full_int_array_732, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_379 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_357 = paddle._C_ops.gather(transpose_12, slice_370, full_379)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_733 = [162]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_734 = [163]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_371 = paddle._C_ops.slice(parameter_48, [0], full_int_array_733, full_int_array_734, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_380 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_358 = paddle._C_ops.gather(transpose_12, slice_371, full_380)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_735 = [163]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_736 = [164]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_372 = paddle._C_ops.slice(parameter_48, [0], full_int_array_735, full_int_array_736, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_381 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_359 = paddle._C_ops.gather(transpose_12, slice_372, full_381)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_737 = [164]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_738 = [165]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_373 = paddle._C_ops.slice(parameter_48, [0], full_int_array_737, full_int_array_738, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_382 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_360 = paddle._C_ops.gather(transpose_12, slice_373, full_382)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_739 = [165]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_740 = [166]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_374 = paddle._C_ops.slice(parameter_48, [0], full_int_array_739, full_int_array_740, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_383 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_361 = paddle._C_ops.gather(transpose_12, slice_374, full_383)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_741 = [166]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_742 = [167]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_375 = paddle._C_ops.slice(parameter_48, [0], full_int_array_741, full_int_array_742, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_384 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_362 = paddle._C_ops.gather(transpose_12, slice_375, full_384)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_743 = [167]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_744 = [168]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_376 = paddle._C_ops.slice(parameter_48, [0], full_int_array_743, full_int_array_744, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_385 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_363 = paddle._C_ops.gather(transpose_12, slice_376, full_385)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_745 = [168]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_746 = [169]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_377 = paddle._C_ops.slice(parameter_48, [0], full_int_array_745, full_int_array_746, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_386 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_364 = paddle._C_ops.gather(transpose_12, slice_377, full_386)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_747 = [169]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_748 = [170]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_378 = paddle._C_ops.slice(parameter_48, [0], full_int_array_747, full_int_array_748, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_387 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_365 = paddle._C_ops.gather(transpose_12, slice_378, full_387)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_749 = [170]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_750 = [171]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_379 = paddle._C_ops.slice(parameter_48, [0], full_int_array_749, full_int_array_750, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_388 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_366 = paddle._C_ops.gather(transpose_12, slice_379, full_388)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_751 = [171]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_752 = [172]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_380 = paddle._C_ops.slice(parameter_48, [0], full_int_array_751, full_int_array_752, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_389 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_367 = paddle._C_ops.gather(transpose_12, slice_380, full_389)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_753 = [172]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_754 = [173]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_381 = paddle._C_ops.slice(parameter_48, [0], full_int_array_753, full_int_array_754, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_390 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_368 = paddle._C_ops.gather(transpose_12, slice_381, full_390)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_755 = [173]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_756 = [174]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_382 = paddle._C_ops.slice(parameter_48, [0], full_int_array_755, full_int_array_756, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_391 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_369 = paddle._C_ops.gather(transpose_12, slice_382, full_391)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_757 = [174]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_758 = [175]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_383 = paddle._C_ops.slice(parameter_48, [0], full_int_array_757, full_int_array_758, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_392 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_370 = paddle._C_ops.gather(transpose_12, slice_383, full_392)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_759 = [175]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_760 = [176]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_384 = paddle._C_ops.slice(parameter_48, [0], full_int_array_759, full_int_array_760, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_393 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_371 = paddle._C_ops.gather(transpose_12, slice_384, full_393)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_761 = [176]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_762 = [177]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_385 = paddle._C_ops.slice(parameter_48, [0], full_int_array_761, full_int_array_762, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_394 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_372 = paddle._C_ops.gather(transpose_12, slice_385, full_394)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_763 = [177]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_764 = [178]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_386 = paddle._C_ops.slice(parameter_48, [0], full_int_array_763, full_int_array_764, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_395 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_373 = paddle._C_ops.gather(transpose_12, slice_386, full_395)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_765 = [178]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_766 = [179]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_387 = paddle._C_ops.slice(parameter_48, [0], full_int_array_765, full_int_array_766, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_396 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_374 = paddle._C_ops.gather(transpose_12, slice_387, full_396)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_767 = [179]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_768 = [180]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_388 = paddle._C_ops.slice(parameter_48, [0], full_int_array_767, full_int_array_768, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_397 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_375 = paddle._C_ops.gather(transpose_12, slice_388, full_397)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_769 = [180]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_770 = [181]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_389 = paddle._C_ops.slice(parameter_48, [0], full_int_array_769, full_int_array_770, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_398 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_376 = paddle._C_ops.gather(transpose_12, slice_389, full_398)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_771 = [181]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_772 = [182]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_390 = paddle._C_ops.slice(parameter_48, [0], full_int_array_771, full_int_array_772, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_399 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_377 = paddle._C_ops.gather(transpose_12, slice_390, full_399)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_773 = [182]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_774 = [183]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_391 = paddle._C_ops.slice(parameter_48, [0], full_int_array_773, full_int_array_774, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_400 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_378 = paddle._C_ops.gather(transpose_12, slice_391, full_400)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_775 = [183]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_776 = [184]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_392 = paddle._C_ops.slice(parameter_48, [0], full_int_array_775, full_int_array_776, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_401 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_379 = paddle._C_ops.gather(transpose_12, slice_392, full_401)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_777 = [184]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_778 = [185]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_393 = paddle._C_ops.slice(parameter_48, [0], full_int_array_777, full_int_array_778, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_402 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_380 = paddle._C_ops.gather(transpose_12, slice_393, full_402)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_779 = [185]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_780 = [186]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_394 = paddle._C_ops.slice(parameter_48, [0], full_int_array_779, full_int_array_780, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_403 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_381 = paddle._C_ops.gather(transpose_12, slice_394, full_403)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_781 = [186]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_782 = [187]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_395 = paddle._C_ops.slice(parameter_48, [0], full_int_array_781, full_int_array_782, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_404 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_382 = paddle._C_ops.gather(transpose_12, slice_395, full_404)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_783 = [187]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_784 = [188]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_396 = paddle._C_ops.slice(parameter_48, [0], full_int_array_783, full_int_array_784, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_405 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_383 = paddle._C_ops.gather(transpose_12, slice_396, full_405)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_785 = [188]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_786 = [189]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_397 = paddle._C_ops.slice(parameter_48, [0], full_int_array_785, full_int_array_786, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_406 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_384 = paddle._C_ops.gather(transpose_12, slice_397, full_406)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_787 = [189]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_788 = [190]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_398 = paddle._C_ops.slice(parameter_48, [0], full_int_array_787, full_int_array_788, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_407 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_385 = paddle._C_ops.gather(transpose_12, slice_398, full_407)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_789 = [190]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_790 = [191]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_399 = paddle._C_ops.slice(parameter_48, [0], full_int_array_789, full_int_array_790, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_408 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_386 = paddle._C_ops.gather(transpose_12, slice_399, full_408)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_791 = [191]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_792 = [192]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_400 = paddle._C_ops.slice(parameter_48, [0], full_int_array_791, full_int_array_792, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_409 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_387 = paddle._C_ops.gather(transpose_12, slice_400, full_409)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_793 = [192]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_794 = [193]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_401 = paddle._C_ops.slice(parameter_48, [0], full_int_array_793, full_int_array_794, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_410 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_388 = paddle._C_ops.gather(transpose_12, slice_401, full_410)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_795 = [193]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_796 = [194]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_402 = paddle._C_ops.slice(parameter_48, [0], full_int_array_795, full_int_array_796, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_411 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_389 = paddle._C_ops.gather(transpose_12, slice_402, full_411)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_797 = [194]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_798 = [195]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_403 = paddle._C_ops.slice(parameter_48, [0], full_int_array_797, full_int_array_798, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_412 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_390 = paddle._C_ops.gather(transpose_12, slice_403, full_412)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_799 = [195]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_800 = [196]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_404 = paddle._C_ops.slice(parameter_48, [0], full_int_array_799, full_int_array_800, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_413 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_391 = paddle._C_ops.gather(transpose_12, slice_404, full_413)

        # builtin.combine: ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32]) <- (196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32)
        combine_9 = [gather_196, gather_197, gather_198, gather_199, gather_200, gather_201, gather_202, gather_203, gather_204, gather_205, gather_206, gather_207, gather_208, gather_209, gather_210, gather_211, gather_212, gather_213, gather_214, gather_215, gather_216, gather_217, gather_218, gather_219, gather_220, gather_221, gather_222, gather_223, gather_224, gather_225, gather_226, gather_227, gather_228, gather_229, gather_230, gather_231, gather_232, gather_233, gather_234, gather_235, gather_236, gather_237, gather_238, gather_239, gather_240, gather_241, gather_242, gather_243, gather_244, gather_245, gather_246, gather_247, gather_248, gather_249, gather_250, gather_251, gather_252, gather_253, gather_254, gather_255, gather_256, gather_257, gather_258, gather_259, gather_260, gather_261, gather_262, gather_263, gather_264, gather_265, gather_266, gather_267, gather_268, gather_269, gather_270, gather_271, gather_272, gather_273, gather_274, gather_275, gather_276, gather_277, gather_278, gather_279, gather_280, gather_281, gather_282, gather_283, gather_284, gather_285, gather_286, gather_287, gather_288, gather_289, gather_290, gather_291, gather_292, gather_293, gather_294, gather_295, gather_296, gather_297, gather_298, gather_299, gather_300, gather_301, gather_302, gather_303, gather_304, gather_305, gather_306, gather_307, gather_308, gather_309, gather_310, gather_311, gather_312, gather_313, gather_314, gather_315, gather_316, gather_317, gather_318, gather_319, gather_320, gather_321, gather_322, gather_323, gather_324, gather_325, gather_326, gather_327, gather_328, gather_329, gather_330, gather_331, gather_332, gather_333, gather_334, gather_335, gather_336, gather_337, gather_338, gather_339, gather_340, gather_341, gather_342, gather_343, gather_344, gather_345, gather_346, gather_347, gather_348, gather_349, gather_350, gather_351, gather_352, gather_353, gather_354, gather_355, gather_356, gather_357, gather_358, gather_359, gather_360, gather_361, gather_362, gather_363, gather_364, gather_365, gather_366, gather_367, gather_368, gather_369, gather_370, gather_371, gather_372, gather_373, gather_374, gather_375, gather_376, gather_377, gather_378, gather_379, gather_380, gather_381, gather_382, gather_383, gather_384, gather_385, gather_386, gather_387, gather_388, gather_389, gather_390, gather_391]

        # pd_op.full: (1xi32) <- ()
        full_414 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (38416x4xf32) <- ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_9, full_414)

        # pd_op.transpose: (4x38416xf32) <- (38416x4xf32)
        transpose_13 = paddle._C_ops.transpose(concat_1, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_801 = [0, 196, 196]

        # pd_op.reshape_: (4x196x196xf32, 0x4x38416xf32) <- (4x38416xf32, 3xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, full_int_array_801), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x4x196x196xf32) <- (-1x4x196x16xf32, -1x4x16x196xf32)
        matmul_7 = paddle._C_ops.matmul(transpose_8, transpose_11, False, False)

        # pd_op.full: (1xf32) <- ()
        full_415 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_7, full_415, float('0'), True)

        # pd_op.add_: (-1x4x196x196xf32) <- (-1x4x196x196xf32, 4x196x196xf32)
        add__3 = paddle._C_ops.add_(scale__1, reshape__18)

        # pd_op.softmax_: (-1x4x196x196xf32) <- (-1x4x196x196xf32)
        softmax__1 = paddle._C_ops.softmax_(add__3, -1)

        # pd_op.matmul: (-1x4x196x32xf32) <- (-1x4x196x196xf32, -1x4x196x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax__1, transpose_10, False, False)

        # pd_op.transpose: (-1x196x4x32xf32) <- (-1x4x196x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_416 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_417 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_204, full_416, full_417]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x196x4x32xf32) <- (-1x196x4x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x128xf32) <- (-1x196x128xf32)
        hardswish_5 = paddle._C_ops.hardswish(reshape__20)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x128xf32, 128x128xf32)
        matmul_9 = paddle._C_ops.matmul(hardswish_5, parameter_49, False, False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_10, flatten_11 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_9, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_10, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_7 = paddle._C_ops.shape(matmul_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_802 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_803 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_405 = paddle._C_ops.slice(shape_7, [0], full_int_array_802, full_int_array_803, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_418 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_419 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_11 = [slice_405, full_418, full_419]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__54, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__4 = paddle._C_ops.add_(add__2, reshape__22)

        # pd_op.matmul: (-1x196x256xf32) <- (-1x196x128xf32, 128x256xf32)
        matmul_10 = paddle._C_ops.matmul(add__4, parameter_54, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x196x256xf32)
        flatten_12, flatten_13 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_10, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_12, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x256xf32)
        shape_8 = paddle._C_ops.shape(matmul_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_804 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_805 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_406 = paddle._C_ops.slice(shape_8, [0], full_int_array_804, full_int_array_805, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_420 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_421 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_406, full_420, full_421]

        # pd_op.reshape_: (-1x196x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__60, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x196x256xf32) <- (-1x196x256xf32)
        hardswish_6 = paddle._C_ops.hardswish(reshape__24)

        # pd_op.matmul: (-1x196x128xf32) <- (-1x196x256xf32, 256x128xf32)
        matmul_11 = paddle._C_ops.matmul(hardswish_6, parameter_59, False, False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x196x128xf32)
        flatten_14, flatten_15 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_11, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_14, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_9 = paddle._C_ops.shape(matmul_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_806 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_807 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_407 = paddle._C_ops.slice(shape_9, [0], full_int_array_806, full_int_array_807, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_422 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_423 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_13 = [slice_407, full_422, full_423]

        # pd_op.reshape_: (-1x196x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__66, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x128xf32) <- (-1x196x128xf32, -1x196x128xf32)
        add__5 = paddle._C_ops.add_(add__4, reshape__26)

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_10 = paddle._C_ops.shape(add__5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_808 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_809 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_408 = paddle._C_ops.slice(shape_10, [0], full_int_array_808, full_int_array_809, [1], [0])

        # pd_op.matmul: (-1x196x640xf32) <- (-1x196x128xf32, 128x640xf32)
        matmul_12 = paddle._C_ops.matmul(add__5, parameter_64, False, False)

        # pd_op.flatten: (-1x640xf32, None) <- (-1x196x640xf32)
        flatten_16, flatten_17 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_12, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x640xf32, 640xf32, 640xf32, xf32, xf32, None) <- (-1x640xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_16, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x196x640xf32)
        shape_11 = paddle._C_ops.shape(matmul_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_810 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_811 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_409 = paddle._C_ops.slice(shape_11, [0], full_int_array_810, full_int_array_811, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_424 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_425 = paddle._C_ops.full([1], float('640'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_14 = [slice_409, full_424, full_425]

        # pd_op.reshape_: (-1x196x640xf32, 0x-1x640xf32) <- (-1x640xf32, [1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__72, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_426 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_427 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_428 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_408, full_426, full_427, full_428]

        # pd_op.reshape_: (-1x196x8x-1xf32, 0x-1x196x640xf32) <- (-1x196x640xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__28, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_812 = [16, 64]

        # pd_op.full: (1xi32) <- ()
        full_429 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x196x8x-1xf32, -1x196x8x-1xf32]) <- (-1x196x8x-1xf32, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(reshape__30, full_int_array_812, full_429)

        # builtin.slice: (-1x196x8x-1xf32) <- ([-1x196x8x-1xf32, -1x196x8x-1xf32])
        slice_410 = split_2[0]

        # pd_op.transpose: (-1x8x196x-1xf32) <- (-1x196x8x-1xf32)
        transpose_15 = paddle._C_ops.transpose(slice_410, [0, 2, 1, 3])

        # builtin.slice: (-1x196x8x-1xf32) <- ([-1x196x8x-1xf32, -1x196x8x-1xf32])
        slice_411 = split_2[1]

        # pd_op.transpose: (-1x8x196x-1xf32) <- (-1x196x8x-1xf32)
        transpose_16 = paddle._C_ops.transpose(slice_411, [0, 2, 1, 3])

        # pd_op.shape: (3xi32) <- (-1x196x128xf32)
        shape_12 = paddle._C_ops.shape(add__5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_813 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_814 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_412 = paddle._C_ops.slice(shape_12, [0], full_int_array_813, full_int_array_814, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_430 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_431 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_432 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_412, full_430, full_431, full_432]

        # pd_op.reshape_: (-1x14x14x128xf32, 0x-1x196x128xf32) <- (-1x196x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__5, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_815 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_816 = [14, 14]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_817 = [2, 2]

        # pd_op.strided_slice: (-1x7x7x128xf32) <- (-1x14x14x128xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(reshape__32, [1, 2], full_int_array_815, full_int_array_816, full_int_array_817)

        # pd_op.full: (1xi32) <- ()
        full_433 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_434 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_412, full_433, full_434]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x7x7x128xf32) <- (-1x7x7x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(strided_slice_0, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_13 = paddle._C_ops.matmul(reshape__34, parameter_69, False, False)

        # pd_op.flatten: (-1x128xf32, None) <- (-1x-1x128xf32)
        flatten_18, flatten_19 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_13, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_18, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x128xf32)
        shape_13 = paddle._C_ops.shape(matmul_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_818 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_819 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_413 = paddle._C_ops.slice(shape_13, [0], full_int_array_818, full_int_array_819, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_820 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_821 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_414 = paddle._C_ops.slice(shape_13, [0], full_int_array_820, full_int_array_821, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_435 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_413, slice_414, full_435]

        # pd_op.reshape_: (-1x-1x128xf32, 0x-1x128xf32) <- (-1x128xf32, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__78, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_436 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_437 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_438 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_408, full_436, full_437, full_438]

        # pd_op.reshape_: (-1x49x8x16xf32, 0x-1x-1x128xf32) <- (-1x-1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__36, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x49x16xf32) <- (-1x49x8x16xf32)
        transpose_17 = paddle._C_ops.transpose(reshape__38, [0, 2, 1, 3])

        # pd_op.transpose: (196x8xf32) <- (8x196xf32)
        transpose_18 = paddle._C_ops.transpose(parameter_74, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_822 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_823 = [1]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_415 = paddle._C_ops.slice(parameter_75, [0], full_int_array_822, full_int_array_823, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_439 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_392 = paddle._C_ops.gather(transpose_18, slice_415, full_439)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_824 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_825 = [2]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_416 = paddle._C_ops.slice(parameter_75, [0], full_int_array_824, full_int_array_825, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_440 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_393 = paddle._C_ops.gather(transpose_18, slice_416, full_440)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_826 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_827 = [3]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_417 = paddle._C_ops.slice(parameter_75, [0], full_int_array_826, full_int_array_827, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_441 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_394 = paddle._C_ops.gather(transpose_18, slice_417, full_441)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_828 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_829 = [4]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_418 = paddle._C_ops.slice(parameter_75, [0], full_int_array_828, full_int_array_829, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_442 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_395 = paddle._C_ops.gather(transpose_18, slice_418, full_442)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_830 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_831 = [5]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_419 = paddle._C_ops.slice(parameter_75, [0], full_int_array_830, full_int_array_831, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_443 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_396 = paddle._C_ops.gather(transpose_18, slice_419, full_443)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_832 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_833 = [6]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_420 = paddle._C_ops.slice(parameter_75, [0], full_int_array_832, full_int_array_833, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_444 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_397 = paddle._C_ops.gather(transpose_18, slice_420, full_444)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_834 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_835 = [7]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_421 = paddle._C_ops.slice(parameter_75, [0], full_int_array_834, full_int_array_835, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_445 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_398 = paddle._C_ops.gather(transpose_18, slice_421, full_445)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_836 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_837 = [8]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_422 = paddle._C_ops.slice(parameter_75, [0], full_int_array_836, full_int_array_837, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_446 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_399 = paddle._C_ops.gather(transpose_18, slice_422, full_446)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_838 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_839 = [9]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_423 = paddle._C_ops.slice(parameter_75, [0], full_int_array_838, full_int_array_839, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_447 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_400 = paddle._C_ops.gather(transpose_18, slice_423, full_447)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_840 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_841 = [10]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_424 = paddle._C_ops.slice(parameter_75, [0], full_int_array_840, full_int_array_841, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_448 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_401 = paddle._C_ops.gather(transpose_18, slice_424, full_448)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_842 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_843 = [11]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_425 = paddle._C_ops.slice(parameter_75, [0], full_int_array_842, full_int_array_843, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_449 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_402 = paddle._C_ops.gather(transpose_18, slice_425, full_449)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_844 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_845 = [12]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_426 = paddle._C_ops.slice(parameter_75, [0], full_int_array_844, full_int_array_845, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_450 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_403 = paddle._C_ops.gather(transpose_18, slice_426, full_450)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_846 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_847 = [13]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_427 = paddle._C_ops.slice(parameter_75, [0], full_int_array_846, full_int_array_847, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_451 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_404 = paddle._C_ops.gather(transpose_18, slice_427, full_451)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_848 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_849 = [14]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_428 = paddle._C_ops.slice(parameter_75, [0], full_int_array_848, full_int_array_849, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_452 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_405 = paddle._C_ops.gather(transpose_18, slice_428, full_452)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_850 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_851 = [15]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_429 = paddle._C_ops.slice(parameter_75, [0], full_int_array_850, full_int_array_851, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_453 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_406 = paddle._C_ops.gather(transpose_18, slice_429, full_453)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_852 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_853 = [16]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_430 = paddle._C_ops.slice(parameter_75, [0], full_int_array_852, full_int_array_853, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_454 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_407 = paddle._C_ops.gather(transpose_18, slice_430, full_454)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_854 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_855 = [17]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_431 = paddle._C_ops.slice(parameter_75, [0], full_int_array_854, full_int_array_855, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_455 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_408 = paddle._C_ops.gather(transpose_18, slice_431, full_455)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_856 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_857 = [18]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_432 = paddle._C_ops.slice(parameter_75, [0], full_int_array_856, full_int_array_857, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_456 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_409 = paddle._C_ops.gather(transpose_18, slice_432, full_456)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_858 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_859 = [19]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_433 = paddle._C_ops.slice(parameter_75, [0], full_int_array_858, full_int_array_859, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_457 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_410 = paddle._C_ops.gather(transpose_18, slice_433, full_457)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_860 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_861 = [20]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_434 = paddle._C_ops.slice(parameter_75, [0], full_int_array_860, full_int_array_861, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_458 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_411 = paddle._C_ops.gather(transpose_18, slice_434, full_458)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_862 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_863 = [21]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_435 = paddle._C_ops.slice(parameter_75, [0], full_int_array_862, full_int_array_863, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_459 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_412 = paddle._C_ops.gather(transpose_18, slice_435, full_459)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_864 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_865 = [22]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_436 = paddle._C_ops.slice(parameter_75, [0], full_int_array_864, full_int_array_865, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_460 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_413 = paddle._C_ops.gather(transpose_18, slice_436, full_460)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_866 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_867 = [23]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_437 = paddle._C_ops.slice(parameter_75, [0], full_int_array_866, full_int_array_867, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_461 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_414 = paddle._C_ops.gather(transpose_18, slice_437, full_461)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_868 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_869 = [24]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_438 = paddle._C_ops.slice(parameter_75, [0], full_int_array_868, full_int_array_869, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_462 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_415 = paddle._C_ops.gather(transpose_18, slice_438, full_462)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_870 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_871 = [25]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_439 = paddle._C_ops.slice(parameter_75, [0], full_int_array_870, full_int_array_871, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_463 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_416 = paddle._C_ops.gather(transpose_18, slice_439, full_463)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_872 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_873 = [26]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_440 = paddle._C_ops.slice(parameter_75, [0], full_int_array_872, full_int_array_873, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_464 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_417 = paddle._C_ops.gather(transpose_18, slice_440, full_464)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_874 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_875 = [27]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_441 = paddle._C_ops.slice(parameter_75, [0], full_int_array_874, full_int_array_875, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_465 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_418 = paddle._C_ops.gather(transpose_18, slice_441, full_465)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_876 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_877 = [28]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_442 = paddle._C_ops.slice(parameter_75, [0], full_int_array_876, full_int_array_877, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_466 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_419 = paddle._C_ops.gather(transpose_18, slice_442, full_466)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_878 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_879 = [29]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_443 = paddle._C_ops.slice(parameter_75, [0], full_int_array_878, full_int_array_879, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_467 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_420 = paddle._C_ops.gather(transpose_18, slice_443, full_467)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_880 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_881 = [30]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_444 = paddle._C_ops.slice(parameter_75, [0], full_int_array_880, full_int_array_881, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_468 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_421 = paddle._C_ops.gather(transpose_18, slice_444, full_468)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_882 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_883 = [31]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_445 = paddle._C_ops.slice(parameter_75, [0], full_int_array_882, full_int_array_883, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_469 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_422 = paddle._C_ops.gather(transpose_18, slice_445, full_469)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_884 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_885 = [32]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_446 = paddle._C_ops.slice(parameter_75, [0], full_int_array_884, full_int_array_885, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_470 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_423 = paddle._C_ops.gather(transpose_18, slice_446, full_470)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_886 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_887 = [33]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_447 = paddle._C_ops.slice(parameter_75, [0], full_int_array_886, full_int_array_887, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_471 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_424 = paddle._C_ops.gather(transpose_18, slice_447, full_471)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_888 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_889 = [34]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_448 = paddle._C_ops.slice(parameter_75, [0], full_int_array_888, full_int_array_889, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_472 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_425 = paddle._C_ops.gather(transpose_18, slice_448, full_472)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_890 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_891 = [35]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_449 = paddle._C_ops.slice(parameter_75, [0], full_int_array_890, full_int_array_891, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_473 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_426 = paddle._C_ops.gather(transpose_18, slice_449, full_473)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_892 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_893 = [36]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_450 = paddle._C_ops.slice(parameter_75, [0], full_int_array_892, full_int_array_893, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_474 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_427 = paddle._C_ops.gather(transpose_18, slice_450, full_474)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_894 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_895 = [37]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_451 = paddle._C_ops.slice(parameter_75, [0], full_int_array_894, full_int_array_895, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_475 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_428 = paddle._C_ops.gather(transpose_18, slice_451, full_475)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_896 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_897 = [38]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_452 = paddle._C_ops.slice(parameter_75, [0], full_int_array_896, full_int_array_897, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_476 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_429 = paddle._C_ops.gather(transpose_18, slice_452, full_476)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_898 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_899 = [39]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_453 = paddle._C_ops.slice(parameter_75, [0], full_int_array_898, full_int_array_899, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_477 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_430 = paddle._C_ops.gather(transpose_18, slice_453, full_477)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_900 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_901 = [40]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_454 = paddle._C_ops.slice(parameter_75, [0], full_int_array_900, full_int_array_901, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_478 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_431 = paddle._C_ops.gather(transpose_18, slice_454, full_478)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_902 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_903 = [41]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_455 = paddle._C_ops.slice(parameter_75, [0], full_int_array_902, full_int_array_903, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_479 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_432 = paddle._C_ops.gather(transpose_18, slice_455, full_479)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_904 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_905 = [42]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_456 = paddle._C_ops.slice(parameter_75, [0], full_int_array_904, full_int_array_905, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_480 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_433 = paddle._C_ops.gather(transpose_18, slice_456, full_480)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_906 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_907 = [43]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_457 = paddle._C_ops.slice(parameter_75, [0], full_int_array_906, full_int_array_907, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_481 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_434 = paddle._C_ops.gather(transpose_18, slice_457, full_481)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_908 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_909 = [44]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_458 = paddle._C_ops.slice(parameter_75, [0], full_int_array_908, full_int_array_909, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_482 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_435 = paddle._C_ops.gather(transpose_18, slice_458, full_482)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_910 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_911 = [45]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_459 = paddle._C_ops.slice(parameter_75, [0], full_int_array_910, full_int_array_911, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_483 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_436 = paddle._C_ops.gather(transpose_18, slice_459, full_483)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_912 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_913 = [46]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_460 = paddle._C_ops.slice(parameter_75, [0], full_int_array_912, full_int_array_913, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_484 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_437 = paddle._C_ops.gather(transpose_18, slice_460, full_484)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_914 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_915 = [47]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_461 = paddle._C_ops.slice(parameter_75, [0], full_int_array_914, full_int_array_915, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_485 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_438 = paddle._C_ops.gather(transpose_18, slice_461, full_485)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_916 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_917 = [48]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_462 = paddle._C_ops.slice(parameter_75, [0], full_int_array_916, full_int_array_917, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_486 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_439 = paddle._C_ops.gather(transpose_18, slice_462, full_486)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_918 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_919 = [49]

        # pd_op.slice: (196xi64) <- (49x196xi64, 1xi64, 1xi64)
        slice_463 = paddle._C_ops.slice(parameter_75, [0], full_int_array_918, full_int_array_919, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_487 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (196x8xf32) <- (196x8xf32, 196xi64, 1xi32)
        gather_440 = paddle._C_ops.gather(transpose_18, slice_463, full_487)

        # builtin.combine: ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32]) <- (196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32)
        combine_20 = [gather_392, gather_393, gather_394, gather_395, gather_396, gather_397, gather_398, gather_399, gather_400, gather_401, gather_402, gather_403, gather_404, gather_405, gather_406, gather_407, gather_408, gather_409, gather_410, gather_411, gather_412, gather_413, gather_414, gather_415, gather_416, gather_417, gather_418, gather_419, gather_420, gather_421, gather_422, gather_423, gather_424, gather_425, gather_426, gather_427, gather_428, gather_429, gather_430, gather_431, gather_432, gather_433, gather_434, gather_435, gather_436, gather_437, gather_438, gather_439, gather_440]

        # pd_op.full: (1xi32) <- ()
        full_488 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (9604x8xf32) <- ([196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32, 196x8xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_20, full_488)

        # pd_op.transpose: (8x9604xf32) <- (9604x8xf32)
        transpose_19 = paddle._C_ops.transpose(concat_2, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_920 = [0, 49, 196]

        # pd_op.reshape_: (8x49x196xf32, 0x8x9604xf32) <- (8x9604xf32, 3xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, full_int_array_920), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x-1x196xf32) <- (-1x8x196x-1xf32)
        transpose_20 = paddle._C_ops.transpose(transpose_15, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x49x196xf32) <- (-1x8x49x16xf32, -1x8x-1x196xf32)
        matmul_14 = paddle._C_ops.matmul(transpose_17, transpose_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_489 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8x49x196xf32) <- (-1x8x49x196xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_14, full_489, float('0'), True)

        # pd_op.add_: (-1x8x49x196xf32) <- (-1x8x49x196xf32, 8x49x196xf32)
        add__6 = paddle._C_ops.add_(scale__2, reshape__40)

        # pd_op.softmax_: (-1x8x49x196xf32) <- (-1x8x49x196xf32)
        softmax__2 = paddle._C_ops.softmax_(add__6, -1)

        # pd_op.matmul: (-1x8x49x-1xf32) <- (-1x8x49x196xf32, -1x8x196x-1xf32)
        matmul_15 = paddle._C_ops.matmul(softmax__2, transpose_16, False, False)

        # pd_op.transpose: (-1x49x8x-1xf32) <- (-1x8x49x-1xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_490 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_491 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_408, full_490, full_491]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x49x8x-1xf32) <- (-1x49x8x-1xf32, [1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_7 = paddle._C_ops.hardswish(reshape__42)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_16 = paddle._C_ops.matmul(hardswish_7, parameter_76, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_20, flatten_21 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_16, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_20, parameter_77, parameter_78, parameter_79, parameter_80, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_14 = paddle._C_ops.shape(matmul_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_921 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_922 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_464 = paddle._C_ops.slice(shape_14, [0], full_int_array_921, full_int_array_922, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_923 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_924 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_465 = paddle._C_ops.slice(shape_14, [0], full_int_array_923, full_int_array_924, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_492 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_464, slice_465, full_492]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__84, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_17 = paddle._C_ops.matmul(reshape__44, parameter_81, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_22, flatten_23 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_17, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_22, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_15 = paddle._C_ops.shape(matmul_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_925 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_926 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_466 = paddle._C_ops.slice(shape_15, [0], full_int_array_925, full_int_array_926, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_927 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_928 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_467 = paddle._C_ops.slice(shape_15, [0], full_int_array_927, full_int_array_928, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_493 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_23 = [slice_466, slice_467, full_493]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__90, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_8 = paddle._C_ops.hardswish(reshape__46)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_18 = paddle._C_ops.matmul(hardswish_8, parameter_86, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_24, flatten_25 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_18, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_24, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_16 = paddle._C_ops.shape(matmul_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_929 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_930 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_468 = paddle._C_ops.slice(shape_16, [0], full_int_array_929, full_int_array_930, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_931 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_932 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_469 = paddle._C_ops.slice(shape_16, [0], full_int_array_931, full_int_array_932, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_494 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_24 = [slice_468, slice_469, full_494]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__96, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_0 = paddle._C_ops.add(reshape__44, reshape__48)

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_17 = paddle._C_ops.shape(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_933 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_934 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_470 = paddle._C_ops.slice(shape_17, [0], full_int_array_933, full_int_array_934, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_935 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_936 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_471 = paddle._C_ops.slice(shape_17, [0], full_int_array_935, full_int_array_936, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_19 = paddle._C_ops.matmul(add_0, parameter_91, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_26, flatten_27 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_19, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_26, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_18 = paddle._C_ops.shape(matmul_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_937 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_938 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_472 = paddle._C_ops.slice(shape_18, [0], full_int_array_937, full_int_array_938, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_939 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_940 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_473 = paddle._C_ops.slice(shape_18, [0], full_int_array_939, full_int_array_940, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_495 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_472, slice_473, full_495]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__102, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_496 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_497 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_470, slice_471, full_496, full_497]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__50, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_941 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_498 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_3 = paddle._C_ops.split(reshape__52, full_int_array_941, full_498)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_474 = split_3[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_22 = paddle._C_ops.transpose(slice_474, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_475 = split_3[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_23 = paddle._C_ops.transpose(slice_475, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_476 = split_3[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_24 = paddle._C_ops.transpose(slice_476, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_25 = paddle._C_ops.transpose(transpose_23, [0, 1, 3, 2])

        # pd_op.transpose: (49x6xf32) <- (6x49xf32)
        transpose_26 = paddle._C_ops.transpose(parameter_96, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_942 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_943 = [1]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_477 = paddle._C_ops.slice(parameter_97, [0], full_int_array_942, full_int_array_943, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_499 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_441 = paddle._C_ops.gather(transpose_26, slice_477, full_499)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_944 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_945 = [2]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_478 = paddle._C_ops.slice(parameter_97, [0], full_int_array_944, full_int_array_945, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_500 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_442 = paddle._C_ops.gather(transpose_26, slice_478, full_500)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_946 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_947 = [3]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_479 = paddle._C_ops.slice(parameter_97, [0], full_int_array_946, full_int_array_947, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_501 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_443 = paddle._C_ops.gather(transpose_26, slice_479, full_501)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_948 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_949 = [4]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_480 = paddle._C_ops.slice(parameter_97, [0], full_int_array_948, full_int_array_949, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_502 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_444 = paddle._C_ops.gather(transpose_26, slice_480, full_502)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_950 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_951 = [5]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_481 = paddle._C_ops.slice(parameter_97, [0], full_int_array_950, full_int_array_951, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_503 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_445 = paddle._C_ops.gather(transpose_26, slice_481, full_503)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_952 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_953 = [6]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_482 = paddle._C_ops.slice(parameter_97, [0], full_int_array_952, full_int_array_953, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_504 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_446 = paddle._C_ops.gather(transpose_26, slice_482, full_504)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_954 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_955 = [7]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_483 = paddle._C_ops.slice(parameter_97, [0], full_int_array_954, full_int_array_955, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_505 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_447 = paddle._C_ops.gather(transpose_26, slice_483, full_505)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_956 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_957 = [8]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_484 = paddle._C_ops.slice(parameter_97, [0], full_int_array_956, full_int_array_957, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_506 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_448 = paddle._C_ops.gather(transpose_26, slice_484, full_506)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_958 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_959 = [9]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_485 = paddle._C_ops.slice(parameter_97, [0], full_int_array_958, full_int_array_959, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_507 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_449 = paddle._C_ops.gather(transpose_26, slice_485, full_507)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_960 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_961 = [10]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_486 = paddle._C_ops.slice(parameter_97, [0], full_int_array_960, full_int_array_961, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_508 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_450 = paddle._C_ops.gather(transpose_26, slice_486, full_508)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_962 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_963 = [11]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_487 = paddle._C_ops.slice(parameter_97, [0], full_int_array_962, full_int_array_963, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_509 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_451 = paddle._C_ops.gather(transpose_26, slice_487, full_509)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_964 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_965 = [12]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_488 = paddle._C_ops.slice(parameter_97, [0], full_int_array_964, full_int_array_965, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_510 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_452 = paddle._C_ops.gather(transpose_26, slice_488, full_510)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_966 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_967 = [13]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_489 = paddle._C_ops.slice(parameter_97, [0], full_int_array_966, full_int_array_967, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_511 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_453 = paddle._C_ops.gather(transpose_26, slice_489, full_511)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_968 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_969 = [14]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_490 = paddle._C_ops.slice(parameter_97, [0], full_int_array_968, full_int_array_969, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_512 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_454 = paddle._C_ops.gather(transpose_26, slice_490, full_512)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_970 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_971 = [15]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_491 = paddle._C_ops.slice(parameter_97, [0], full_int_array_970, full_int_array_971, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_513 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_455 = paddle._C_ops.gather(transpose_26, slice_491, full_513)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_972 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_973 = [16]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_492 = paddle._C_ops.slice(parameter_97, [0], full_int_array_972, full_int_array_973, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_514 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_456 = paddle._C_ops.gather(transpose_26, slice_492, full_514)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_974 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_975 = [17]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_493 = paddle._C_ops.slice(parameter_97, [0], full_int_array_974, full_int_array_975, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_515 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_457 = paddle._C_ops.gather(transpose_26, slice_493, full_515)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_976 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_977 = [18]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_494 = paddle._C_ops.slice(parameter_97, [0], full_int_array_976, full_int_array_977, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_516 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_458 = paddle._C_ops.gather(transpose_26, slice_494, full_516)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_978 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_979 = [19]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_495 = paddle._C_ops.slice(parameter_97, [0], full_int_array_978, full_int_array_979, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_517 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_459 = paddle._C_ops.gather(transpose_26, slice_495, full_517)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_980 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_981 = [20]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_496 = paddle._C_ops.slice(parameter_97, [0], full_int_array_980, full_int_array_981, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_518 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_460 = paddle._C_ops.gather(transpose_26, slice_496, full_518)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_982 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_983 = [21]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_497 = paddle._C_ops.slice(parameter_97, [0], full_int_array_982, full_int_array_983, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_519 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_461 = paddle._C_ops.gather(transpose_26, slice_497, full_519)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_984 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_985 = [22]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_498 = paddle._C_ops.slice(parameter_97, [0], full_int_array_984, full_int_array_985, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_520 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_462 = paddle._C_ops.gather(transpose_26, slice_498, full_520)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_986 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_987 = [23]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_499 = paddle._C_ops.slice(parameter_97, [0], full_int_array_986, full_int_array_987, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_521 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_463 = paddle._C_ops.gather(transpose_26, slice_499, full_521)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_988 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_989 = [24]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_500 = paddle._C_ops.slice(parameter_97, [0], full_int_array_988, full_int_array_989, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_522 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_464 = paddle._C_ops.gather(transpose_26, slice_500, full_522)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_990 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_991 = [25]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_501 = paddle._C_ops.slice(parameter_97, [0], full_int_array_990, full_int_array_991, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_523 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_465 = paddle._C_ops.gather(transpose_26, slice_501, full_523)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_992 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_993 = [26]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_502 = paddle._C_ops.slice(parameter_97, [0], full_int_array_992, full_int_array_993, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_524 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_466 = paddle._C_ops.gather(transpose_26, slice_502, full_524)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_994 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_995 = [27]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_503 = paddle._C_ops.slice(parameter_97, [0], full_int_array_994, full_int_array_995, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_525 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_467 = paddle._C_ops.gather(transpose_26, slice_503, full_525)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_996 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_997 = [28]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_504 = paddle._C_ops.slice(parameter_97, [0], full_int_array_996, full_int_array_997, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_526 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_468 = paddle._C_ops.gather(transpose_26, slice_504, full_526)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_998 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_999 = [29]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_505 = paddle._C_ops.slice(parameter_97, [0], full_int_array_998, full_int_array_999, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_527 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_469 = paddle._C_ops.gather(transpose_26, slice_505, full_527)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1000 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1001 = [30]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_506 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1000, full_int_array_1001, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_528 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_470 = paddle._C_ops.gather(transpose_26, slice_506, full_528)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1002 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1003 = [31]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_507 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1002, full_int_array_1003, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_529 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_471 = paddle._C_ops.gather(transpose_26, slice_507, full_529)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1004 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1005 = [32]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_508 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1004, full_int_array_1005, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_530 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_472 = paddle._C_ops.gather(transpose_26, slice_508, full_530)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1006 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1007 = [33]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_509 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1006, full_int_array_1007, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_531 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_473 = paddle._C_ops.gather(transpose_26, slice_509, full_531)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1008 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1009 = [34]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_510 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1008, full_int_array_1009, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_532 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_474 = paddle._C_ops.gather(transpose_26, slice_510, full_532)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1010 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1011 = [35]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_511 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1010, full_int_array_1011, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_533 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_475 = paddle._C_ops.gather(transpose_26, slice_511, full_533)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1012 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1013 = [36]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_512 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1012, full_int_array_1013, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_534 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_476 = paddle._C_ops.gather(transpose_26, slice_512, full_534)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1014 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1015 = [37]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_513 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1014, full_int_array_1015, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_535 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_477 = paddle._C_ops.gather(transpose_26, slice_513, full_535)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1016 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1017 = [38]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_514 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1016, full_int_array_1017, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_536 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_478 = paddle._C_ops.gather(transpose_26, slice_514, full_536)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1018 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1019 = [39]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_515 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1018, full_int_array_1019, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_537 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_479 = paddle._C_ops.gather(transpose_26, slice_515, full_537)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1020 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1021 = [40]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_516 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1020, full_int_array_1021, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_538 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_480 = paddle._C_ops.gather(transpose_26, slice_516, full_538)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1022 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1023 = [41]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_517 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1022, full_int_array_1023, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_539 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_481 = paddle._C_ops.gather(transpose_26, slice_517, full_539)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1024 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1025 = [42]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_518 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1024, full_int_array_1025, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_540 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_482 = paddle._C_ops.gather(transpose_26, slice_518, full_540)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1026 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1027 = [43]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_519 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1026, full_int_array_1027, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_541 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_483 = paddle._C_ops.gather(transpose_26, slice_519, full_541)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1028 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1029 = [44]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_520 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1028, full_int_array_1029, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_542 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_484 = paddle._C_ops.gather(transpose_26, slice_520, full_542)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1030 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1031 = [45]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_521 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1030, full_int_array_1031, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_543 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_485 = paddle._C_ops.gather(transpose_26, slice_521, full_543)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1032 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1033 = [46]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_522 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1032, full_int_array_1033, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_544 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_486 = paddle._C_ops.gather(transpose_26, slice_522, full_544)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1034 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1035 = [47]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_523 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1034, full_int_array_1035, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_545 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_487 = paddle._C_ops.gather(transpose_26, slice_523, full_545)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1036 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1037 = [48]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_524 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1036, full_int_array_1037, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_546 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_488 = paddle._C_ops.gather(transpose_26, slice_524, full_546)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1038 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1039 = [49]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_525 = paddle._C_ops.slice(parameter_97, [0], full_int_array_1038, full_int_array_1039, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_547 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_489 = paddle._C_ops.gather(transpose_26, slice_525, full_547)

        # builtin.combine: ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32]) <- (49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32)
        combine_27 = [gather_441, gather_442, gather_443, gather_444, gather_445, gather_446, gather_447, gather_448, gather_449, gather_450, gather_451, gather_452, gather_453, gather_454, gather_455, gather_456, gather_457, gather_458, gather_459, gather_460, gather_461, gather_462, gather_463, gather_464, gather_465, gather_466, gather_467, gather_468, gather_469, gather_470, gather_471, gather_472, gather_473, gather_474, gather_475, gather_476, gather_477, gather_478, gather_479, gather_480, gather_481, gather_482, gather_483, gather_484, gather_485, gather_486, gather_487, gather_488, gather_489]

        # pd_op.full: (1xi32) <- ()
        full_548 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (2401x6xf32) <- ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_27, full_548)

        # pd_op.transpose: (6x2401xf32) <- (2401x6xf32)
        transpose_27 = paddle._C_ops.transpose(concat_3, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1040 = [0, 49, 49]

        # pd_op.reshape_: (6x49x49xf32, 0x6x2401xf32) <- (6x2401xf32, 3xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, full_int_array_1040), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_22, transpose_25, False, False)

        # pd_op.full: (1xf32) <- ()
        full_549 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_20, full_549, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_1 = paddle._C_ops.add(scale_0, reshape__54)

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__3 = paddle._C_ops.softmax_(add_1, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax__3, transpose_24, False, False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_28 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_550 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_470, slice_471, full_550]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_9 = paddle._C_ops.hardswish(reshape__56)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_22 = paddle._C_ops.matmul(hardswish_9, parameter_98, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_28, flatten_29 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_22, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_28, parameter_99, parameter_100, parameter_101, parameter_102, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_19 = paddle._C_ops.shape(matmul_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1041 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1042 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_526 = paddle._C_ops.slice(shape_19, [0], full_int_array_1041, full_int_array_1042, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1043 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1044 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_527 = paddle._C_ops.slice(shape_19, [0], full_int_array_1043, full_int_array_1044, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_551 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_29 = [slice_526, slice_527, full_551]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__108, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_2 = paddle._C_ops.add(add_0, reshape__58)

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_23 = paddle._C_ops.matmul(add_2, parameter_103, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_30, flatten_31 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_23, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_30, parameter_104, parameter_105, parameter_106, parameter_107, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_20 = paddle._C_ops.shape(matmul_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1045 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1046 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_528 = paddle._C_ops.slice(shape_20, [0], full_int_array_1045, full_int_array_1046, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1047 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1048 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_529 = paddle._C_ops.slice(shape_20, [0], full_int_array_1047, full_int_array_1048, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_552 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_30 = [slice_528, slice_529, full_552]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__114, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_10 = paddle._C_ops.hardswish(reshape__60)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_24 = paddle._C_ops.matmul(hardswish_10, parameter_108, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_32, flatten_33 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_24, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_32, parameter_109, parameter_110, parameter_111, parameter_112, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_21 = paddle._C_ops.shape(matmul_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1049 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1050 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_530 = paddle._C_ops.slice(shape_21, [0], full_int_array_1049, full_int_array_1050, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1051 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1052 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_531 = paddle._C_ops.slice(shape_21, [0], full_int_array_1051, full_int_array_1052, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_553 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_31 = [slice_530, slice_531, full_553]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__120, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_3 = paddle._C_ops.add(add_2, reshape__62)

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_22 = paddle._C_ops.shape(add_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1053 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1054 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_532 = paddle._C_ops.slice(shape_22, [0], full_int_array_1053, full_int_array_1054, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1055 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1056 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_533 = paddle._C_ops.slice(shape_22, [0], full_int_array_1055, full_int_array_1056, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_25 = paddle._C_ops.matmul(add_3, parameter_113, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_34, flatten_35 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_25, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_34, parameter_114, parameter_115, parameter_116, parameter_117, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_23 = paddle._C_ops.shape(matmul_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1057 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1058 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_534 = paddle._C_ops.slice(shape_23, [0], full_int_array_1057, full_int_array_1058, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1059 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1060 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_535 = paddle._C_ops.slice(shape_23, [0], full_int_array_1059, full_int_array_1060, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_554 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_534, slice_535, full_554]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__126, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_555 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_556 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_532, slice_533, full_555, full_556]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__64, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1061 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_557 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape__66, full_int_array_1061, full_557)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_536 = split_4[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_29 = paddle._C_ops.transpose(slice_536, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_537 = split_4[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_30 = paddle._C_ops.transpose(slice_537, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_538 = split_4[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_31 = paddle._C_ops.transpose(slice_538, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_32 = paddle._C_ops.transpose(transpose_30, [0, 1, 3, 2])

        # pd_op.transpose: (49x6xf32) <- (6x49xf32)
        transpose_33 = paddle._C_ops.transpose(parameter_118, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1062 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1063 = [1]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_539 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1062, full_int_array_1063, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_558 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_490 = paddle._C_ops.gather(transpose_33, slice_539, full_558)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1064 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1065 = [2]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_540 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1064, full_int_array_1065, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_559 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_491 = paddle._C_ops.gather(transpose_33, slice_540, full_559)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1066 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1067 = [3]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_541 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1066, full_int_array_1067, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_560 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_492 = paddle._C_ops.gather(transpose_33, slice_541, full_560)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1068 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1069 = [4]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_542 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1068, full_int_array_1069, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_561 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_493 = paddle._C_ops.gather(transpose_33, slice_542, full_561)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1070 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1071 = [5]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_543 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1070, full_int_array_1071, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_562 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_494 = paddle._C_ops.gather(transpose_33, slice_543, full_562)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1072 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1073 = [6]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_544 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1072, full_int_array_1073, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_563 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_495 = paddle._C_ops.gather(transpose_33, slice_544, full_563)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1074 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1075 = [7]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_545 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1074, full_int_array_1075, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_564 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_496 = paddle._C_ops.gather(transpose_33, slice_545, full_564)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1076 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1077 = [8]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_546 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1076, full_int_array_1077, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_565 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_497 = paddle._C_ops.gather(transpose_33, slice_546, full_565)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1078 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1079 = [9]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_547 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1078, full_int_array_1079, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_566 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_498 = paddle._C_ops.gather(transpose_33, slice_547, full_566)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1080 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1081 = [10]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_548 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1080, full_int_array_1081, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_567 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_499 = paddle._C_ops.gather(transpose_33, slice_548, full_567)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1082 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1083 = [11]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_549 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1082, full_int_array_1083, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_568 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_500 = paddle._C_ops.gather(transpose_33, slice_549, full_568)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1084 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1085 = [12]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_550 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1084, full_int_array_1085, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_569 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_501 = paddle._C_ops.gather(transpose_33, slice_550, full_569)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1086 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1087 = [13]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_551 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1086, full_int_array_1087, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_570 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_502 = paddle._C_ops.gather(transpose_33, slice_551, full_570)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1088 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1089 = [14]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_552 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1088, full_int_array_1089, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_571 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_503 = paddle._C_ops.gather(transpose_33, slice_552, full_571)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1090 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1091 = [15]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_553 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1090, full_int_array_1091, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_572 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_504 = paddle._C_ops.gather(transpose_33, slice_553, full_572)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1092 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1093 = [16]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_554 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1092, full_int_array_1093, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_573 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_505 = paddle._C_ops.gather(transpose_33, slice_554, full_573)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1094 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1095 = [17]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_555 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1094, full_int_array_1095, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_574 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_506 = paddle._C_ops.gather(transpose_33, slice_555, full_574)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1096 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1097 = [18]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_556 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1096, full_int_array_1097, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_575 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_507 = paddle._C_ops.gather(transpose_33, slice_556, full_575)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1098 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1099 = [19]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_557 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1098, full_int_array_1099, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_576 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_508 = paddle._C_ops.gather(transpose_33, slice_557, full_576)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1100 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1101 = [20]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_558 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1100, full_int_array_1101, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_577 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_509 = paddle._C_ops.gather(transpose_33, slice_558, full_577)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1102 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1103 = [21]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_559 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1102, full_int_array_1103, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_578 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_510 = paddle._C_ops.gather(transpose_33, slice_559, full_578)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1104 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1105 = [22]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_560 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1104, full_int_array_1105, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_579 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_511 = paddle._C_ops.gather(transpose_33, slice_560, full_579)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1106 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1107 = [23]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_561 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1106, full_int_array_1107, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_580 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_512 = paddle._C_ops.gather(transpose_33, slice_561, full_580)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1108 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1109 = [24]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_562 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1108, full_int_array_1109, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_581 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_513 = paddle._C_ops.gather(transpose_33, slice_562, full_581)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1110 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1111 = [25]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_563 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1110, full_int_array_1111, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_582 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_514 = paddle._C_ops.gather(transpose_33, slice_563, full_582)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1112 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1113 = [26]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_564 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1112, full_int_array_1113, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_583 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_515 = paddle._C_ops.gather(transpose_33, slice_564, full_583)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1114 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1115 = [27]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_565 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1114, full_int_array_1115, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_584 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_516 = paddle._C_ops.gather(transpose_33, slice_565, full_584)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1116 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1117 = [28]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_566 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1116, full_int_array_1117, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_585 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_517 = paddle._C_ops.gather(transpose_33, slice_566, full_585)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1118 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1119 = [29]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_567 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1118, full_int_array_1119, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_586 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_518 = paddle._C_ops.gather(transpose_33, slice_567, full_586)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1120 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1121 = [30]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_568 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1120, full_int_array_1121, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_587 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_519 = paddle._C_ops.gather(transpose_33, slice_568, full_587)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1122 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1123 = [31]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_569 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1122, full_int_array_1123, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_588 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_520 = paddle._C_ops.gather(transpose_33, slice_569, full_588)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1124 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1125 = [32]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_570 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1124, full_int_array_1125, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_589 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_521 = paddle._C_ops.gather(transpose_33, slice_570, full_589)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1126 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1127 = [33]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_571 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1126, full_int_array_1127, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_590 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_522 = paddle._C_ops.gather(transpose_33, slice_571, full_590)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1128 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1129 = [34]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_572 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1128, full_int_array_1129, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_591 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_523 = paddle._C_ops.gather(transpose_33, slice_572, full_591)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1130 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1131 = [35]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_573 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1130, full_int_array_1131, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_592 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_524 = paddle._C_ops.gather(transpose_33, slice_573, full_592)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1132 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1133 = [36]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_574 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1132, full_int_array_1133, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_593 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_525 = paddle._C_ops.gather(transpose_33, slice_574, full_593)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1134 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1135 = [37]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_575 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1134, full_int_array_1135, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_594 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_526 = paddle._C_ops.gather(transpose_33, slice_575, full_594)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1136 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1137 = [38]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_576 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1136, full_int_array_1137, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_595 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_527 = paddle._C_ops.gather(transpose_33, slice_576, full_595)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1138 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1139 = [39]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_577 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1138, full_int_array_1139, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_596 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_528 = paddle._C_ops.gather(transpose_33, slice_577, full_596)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1140 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1141 = [40]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_578 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1140, full_int_array_1141, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_597 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_529 = paddle._C_ops.gather(transpose_33, slice_578, full_597)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1142 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1143 = [41]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_579 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1142, full_int_array_1143, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_598 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_530 = paddle._C_ops.gather(transpose_33, slice_579, full_598)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1144 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1145 = [42]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_580 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1144, full_int_array_1145, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_599 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_531 = paddle._C_ops.gather(transpose_33, slice_580, full_599)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1146 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1147 = [43]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_581 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1146, full_int_array_1147, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_600 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_532 = paddle._C_ops.gather(transpose_33, slice_581, full_600)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1148 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1149 = [44]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_582 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1148, full_int_array_1149, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_601 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_533 = paddle._C_ops.gather(transpose_33, slice_582, full_601)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1150 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1151 = [45]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_583 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1150, full_int_array_1151, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_602 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_534 = paddle._C_ops.gather(transpose_33, slice_583, full_602)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1152 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1153 = [46]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_584 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1152, full_int_array_1153, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_603 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_535 = paddle._C_ops.gather(transpose_33, slice_584, full_603)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1154 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1155 = [47]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_585 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1154, full_int_array_1155, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_604 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_536 = paddle._C_ops.gather(transpose_33, slice_585, full_604)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1156 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1157 = [48]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_586 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1156, full_int_array_1157, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_605 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_537 = paddle._C_ops.gather(transpose_33, slice_586, full_605)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1158 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1159 = [49]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_587 = paddle._C_ops.slice(parameter_119, [0], full_int_array_1158, full_int_array_1159, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_606 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_538 = paddle._C_ops.gather(transpose_33, slice_587, full_606)

        # builtin.combine: ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32]) <- (49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32)
        combine_34 = [gather_490, gather_491, gather_492, gather_493, gather_494, gather_495, gather_496, gather_497, gather_498, gather_499, gather_500, gather_501, gather_502, gather_503, gather_504, gather_505, gather_506, gather_507, gather_508, gather_509, gather_510, gather_511, gather_512, gather_513, gather_514, gather_515, gather_516, gather_517, gather_518, gather_519, gather_520, gather_521, gather_522, gather_523, gather_524, gather_525, gather_526, gather_527, gather_528, gather_529, gather_530, gather_531, gather_532, gather_533, gather_534, gather_535, gather_536, gather_537, gather_538]

        # pd_op.full: (1xi32) <- ()
        full_607 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (2401x6xf32) <- ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_34, full_607)

        # pd_op.transpose: (6x2401xf32) <- (2401x6xf32)
        transpose_34 = paddle._C_ops.transpose(concat_4, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1160 = [0, 49, 49]

        # pd_op.reshape_: (6x49x49xf32, 0x6x2401xf32) <- (6x2401xf32, 3xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_34, full_int_array_1160), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_26 = paddle._C_ops.matmul(transpose_29, transpose_32, False, False)

        # pd_op.full: (1xf32) <- ()
        full_608 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_26, full_608, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_4 = paddle._C_ops.add(scale_1, reshape__68)

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__4 = paddle._C_ops.softmax_(add_4, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_27 = paddle._C_ops.matmul(softmax__4, transpose_31, False, False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_609 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_35 = [slice_532, slice_533, full_609]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_35, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_11 = paddle._C_ops.hardswish(reshape__70)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_28 = paddle._C_ops.matmul(hardswish_11, parameter_120, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_36, flatten_37 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_28, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_36, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_24 = paddle._C_ops.shape(matmul_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1161 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1162 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_588 = paddle._C_ops.slice(shape_24, [0], full_int_array_1161, full_int_array_1162, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1163 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1164 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_589 = paddle._C_ops.slice(shape_24, [0], full_int_array_1163, full_int_array_1164, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_610 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_588, slice_589, full_610]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__132, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_5 = paddle._C_ops.add(add_3, reshape__72)

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_29 = paddle._C_ops.matmul(add_5, parameter_125, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_38, flatten_39 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_29, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_38, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_25 = paddle._C_ops.shape(matmul_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1165 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1166 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_590 = paddle._C_ops.slice(shape_25, [0], full_int_array_1165, full_int_array_1166, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1167 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1168 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_591 = paddle._C_ops.slice(shape_25, [0], full_int_array_1167, full_int_array_1168, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_611 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_37 = [slice_590, slice_591, full_611]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__138, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_12 = paddle._C_ops.hardswish(reshape__74)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_30 = paddle._C_ops.matmul(hardswish_12, parameter_130, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_40, flatten_41 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_30, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_40, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_26 = paddle._C_ops.shape(matmul_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1169 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1170 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_592 = paddle._C_ops.slice(shape_26, [0], full_int_array_1169, full_int_array_1170, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1171 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1172 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_593 = paddle._C_ops.slice(shape_26, [0], full_int_array_1171, full_int_array_1172, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_612 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_38 = [slice_592, slice_593, full_612]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__144, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_6 = paddle._C_ops.add(add_5, reshape__76)

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_27 = paddle._C_ops.shape(add_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1173 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1174 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_594 = paddle._C_ops.slice(shape_27, [0], full_int_array_1173, full_int_array_1174, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1175 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1176 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_595 = paddle._C_ops.slice(shape_27, [0], full_int_array_1175, full_int_array_1176, [1], [0])

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_31 = paddle._C_ops.matmul(add_6, parameter_135, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_42, flatten_43 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_31, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_42, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_28 = paddle._C_ops.shape(matmul_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1177 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1178 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_596 = paddle._C_ops.slice(shape_28, [0], full_int_array_1177, full_int_array_1178, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1179 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1180 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_597 = paddle._C_ops.slice(shape_28, [0], full_int_array_1179, full_int_array_1180, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_613 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_39 = [slice_596, slice_597, full_613]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__150, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_614 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_615 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_594, slice_595, full_614, full_615]

        # pd_op.reshape_: (-1x-1x6x64xf32, 0x-1x-1x384xf32) <- (-1x-1x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__78, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1181 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_616 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32]) <- (-1x-1x6x64xf32, 3xi64, 1xi32)
        split_5 = paddle._C_ops.split(reshape__80, full_int_array_1181, full_616)

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_598 = split_5[0]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_36 = paddle._C_ops.transpose(slice_598, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x16xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_599 = split_5[1]

        # pd_op.transpose: (-1x6x-1x16xf32) <- (-1x-1x6x16xf32)
        transpose_37 = paddle._C_ops.transpose(slice_599, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x6x32xf32) <- ([-1x-1x6x16xf32, -1x-1x6x16xf32, -1x-1x6x32xf32])
        slice_600 = split_5[2]

        # pd_op.transpose: (-1x6x-1x32xf32) <- (-1x-1x6x32xf32)
        transpose_38 = paddle._C_ops.transpose(slice_600, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x16x-1xf32) <- (-1x6x-1x16xf32)
        transpose_39 = paddle._C_ops.transpose(transpose_37, [0, 1, 3, 2])

        # pd_op.transpose: (49x6xf32) <- (6x49xf32)
        transpose_40 = paddle._C_ops.transpose(parameter_140, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1182 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1183 = [1]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_601 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1182, full_int_array_1183, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_617 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_539 = paddle._C_ops.gather(transpose_40, slice_601, full_617)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1184 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1185 = [2]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_602 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1184, full_int_array_1185, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_618 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_540 = paddle._C_ops.gather(transpose_40, slice_602, full_618)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1186 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1187 = [3]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_603 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1186, full_int_array_1187, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_619 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_541 = paddle._C_ops.gather(transpose_40, slice_603, full_619)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1188 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1189 = [4]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_604 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1188, full_int_array_1189, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_620 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_542 = paddle._C_ops.gather(transpose_40, slice_604, full_620)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1190 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1191 = [5]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_605 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1190, full_int_array_1191, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_621 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_543 = paddle._C_ops.gather(transpose_40, slice_605, full_621)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1192 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1193 = [6]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_606 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1192, full_int_array_1193, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_622 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_544 = paddle._C_ops.gather(transpose_40, slice_606, full_622)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1194 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1195 = [7]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_607 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1194, full_int_array_1195, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_623 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_545 = paddle._C_ops.gather(transpose_40, slice_607, full_623)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1196 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1197 = [8]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_608 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1196, full_int_array_1197, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_624 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_546 = paddle._C_ops.gather(transpose_40, slice_608, full_624)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1198 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1199 = [9]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_609 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1198, full_int_array_1199, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_625 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_547 = paddle._C_ops.gather(transpose_40, slice_609, full_625)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1200 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1201 = [10]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_610 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1200, full_int_array_1201, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_626 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_548 = paddle._C_ops.gather(transpose_40, slice_610, full_626)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1202 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1203 = [11]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_611 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1202, full_int_array_1203, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_627 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_549 = paddle._C_ops.gather(transpose_40, slice_611, full_627)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1204 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1205 = [12]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_612 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1204, full_int_array_1205, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_628 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_550 = paddle._C_ops.gather(transpose_40, slice_612, full_628)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1206 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1207 = [13]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_613 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1206, full_int_array_1207, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_629 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_551 = paddle._C_ops.gather(transpose_40, slice_613, full_629)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1208 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1209 = [14]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_614 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1208, full_int_array_1209, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_630 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_552 = paddle._C_ops.gather(transpose_40, slice_614, full_630)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1210 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1211 = [15]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_615 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1210, full_int_array_1211, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_631 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_553 = paddle._C_ops.gather(transpose_40, slice_615, full_631)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1212 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1213 = [16]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_616 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1212, full_int_array_1213, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_632 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_554 = paddle._C_ops.gather(transpose_40, slice_616, full_632)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1214 = [16]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1215 = [17]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_617 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1214, full_int_array_1215, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_633 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_555 = paddle._C_ops.gather(transpose_40, slice_617, full_633)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1216 = [17]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1217 = [18]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_618 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1216, full_int_array_1217, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_634 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_556 = paddle._C_ops.gather(transpose_40, slice_618, full_634)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1218 = [18]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1219 = [19]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_619 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1218, full_int_array_1219, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_635 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_557 = paddle._C_ops.gather(transpose_40, slice_619, full_635)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1220 = [19]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1221 = [20]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_620 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1220, full_int_array_1221, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_636 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_558 = paddle._C_ops.gather(transpose_40, slice_620, full_636)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1222 = [20]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1223 = [21]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_621 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1222, full_int_array_1223, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_637 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_559 = paddle._C_ops.gather(transpose_40, slice_621, full_637)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1224 = [21]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1225 = [22]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_622 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1224, full_int_array_1225, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_638 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_560 = paddle._C_ops.gather(transpose_40, slice_622, full_638)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1226 = [22]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1227 = [23]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_623 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1226, full_int_array_1227, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_639 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_561 = paddle._C_ops.gather(transpose_40, slice_623, full_639)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1228 = [23]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1229 = [24]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_624 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1228, full_int_array_1229, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_640 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_562 = paddle._C_ops.gather(transpose_40, slice_624, full_640)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1230 = [24]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1231 = [25]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_625 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1230, full_int_array_1231, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_641 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_563 = paddle._C_ops.gather(transpose_40, slice_625, full_641)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1232 = [25]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1233 = [26]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_626 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1232, full_int_array_1233, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_642 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_564 = paddle._C_ops.gather(transpose_40, slice_626, full_642)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1234 = [26]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1235 = [27]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_627 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1234, full_int_array_1235, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_643 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_565 = paddle._C_ops.gather(transpose_40, slice_627, full_643)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1236 = [27]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1237 = [28]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_628 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1236, full_int_array_1237, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_644 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_566 = paddle._C_ops.gather(transpose_40, slice_628, full_644)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1238 = [28]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1239 = [29]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_629 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1238, full_int_array_1239, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_645 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_567 = paddle._C_ops.gather(transpose_40, slice_629, full_645)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1240 = [29]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1241 = [30]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_630 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1240, full_int_array_1241, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_646 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_568 = paddle._C_ops.gather(transpose_40, slice_630, full_646)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1242 = [30]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1243 = [31]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_631 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1242, full_int_array_1243, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_647 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_569 = paddle._C_ops.gather(transpose_40, slice_631, full_647)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1244 = [31]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1245 = [32]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_632 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1244, full_int_array_1245, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_648 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_570 = paddle._C_ops.gather(transpose_40, slice_632, full_648)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1246 = [32]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1247 = [33]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_633 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1246, full_int_array_1247, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_649 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_571 = paddle._C_ops.gather(transpose_40, slice_633, full_649)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1248 = [33]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1249 = [34]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_634 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1248, full_int_array_1249, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_650 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_572 = paddle._C_ops.gather(transpose_40, slice_634, full_650)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1250 = [34]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1251 = [35]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_635 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1250, full_int_array_1251, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_651 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_573 = paddle._C_ops.gather(transpose_40, slice_635, full_651)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1252 = [35]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1253 = [36]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_636 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1252, full_int_array_1253, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_652 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_574 = paddle._C_ops.gather(transpose_40, slice_636, full_652)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1254 = [36]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1255 = [37]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_637 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1254, full_int_array_1255, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_653 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_575 = paddle._C_ops.gather(transpose_40, slice_637, full_653)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1256 = [37]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1257 = [38]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_638 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1256, full_int_array_1257, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_654 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_576 = paddle._C_ops.gather(transpose_40, slice_638, full_654)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1258 = [38]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1259 = [39]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_639 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1258, full_int_array_1259, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_655 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_577 = paddle._C_ops.gather(transpose_40, slice_639, full_655)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1260 = [39]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1261 = [40]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_640 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1260, full_int_array_1261, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_656 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_578 = paddle._C_ops.gather(transpose_40, slice_640, full_656)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1262 = [40]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1263 = [41]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_641 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1262, full_int_array_1263, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_657 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_579 = paddle._C_ops.gather(transpose_40, slice_641, full_657)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1264 = [41]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1265 = [42]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_642 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1264, full_int_array_1265, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_658 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_580 = paddle._C_ops.gather(transpose_40, slice_642, full_658)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1266 = [42]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1267 = [43]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_643 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1266, full_int_array_1267, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_659 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_581 = paddle._C_ops.gather(transpose_40, slice_643, full_659)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1268 = [43]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1269 = [44]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_644 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1268, full_int_array_1269, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_660 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_582 = paddle._C_ops.gather(transpose_40, slice_644, full_660)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1270 = [44]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1271 = [45]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_645 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1270, full_int_array_1271, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_661 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_583 = paddle._C_ops.gather(transpose_40, slice_645, full_661)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1272 = [45]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1273 = [46]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_646 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1272, full_int_array_1273, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_662 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_584 = paddle._C_ops.gather(transpose_40, slice_646, full_662)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1274 = [46]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1275 = [47]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_647 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1274, full_int_array_1275, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_663 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_585 = paddle._C_ops.gather(transpose_40, slice_647, full_663)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1276 = [47]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1277 = [48]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_648 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1276, full_int_array_1277, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_664 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_586 = paddle._C_ops.gather(transpose_40, slice_648, full_664)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1278 = [48]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1279 = [49]

        # pd_op.slice: (49xi64) <- (49x49xi64, 1xi64, 1xi64)
        slice_649 = paddle._C_ops.slice(parameter_141, [0], full_int_array_1278, full_int_array_1279, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_665 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x6xf32) <- (49x6xf32, 49xi64, 1xi32)
        gather_587 = paddle._C_ops.gather(transpose_40, slice_649, full_665)

        # builtin.combine: ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32]) <- (49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32)
        combine_41 = [gather_539, gather_540, gather_541, gather_542, gather_543, gather_544, gather_545, gather_546, gather_547, gather_548, gather_549, gather_550, gather_551, gather_552, gather_553, gather_554, gather_555, gather_556, gather_557, gather_558, gather_559, gather_560, gather_561, gather_562, gather_563, gather_564, gather_565, gather_566, gather_567, gather_568, gather_569, gather_570, gather_571, gather_572, gather_573, gather_574, gather_575, gather_576, gather_577, gather_578, gather_579, gather_580, gather_581, gather_582, gather_583, gather_584, gather_585, gather_586, gather_587]

        # pd_op.full: (1xi32) <- ()
        full_666 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (2401x6xf32) <- ([49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32, 49x6xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_41, full_666)

        # pd_op.transpose: (6x2401xf32) <- (2401x6xf32)
        transpose_41 = paddle._C_ops.transpose(concat_5, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1280 = [0, 49, 49]

        # pd_op.reshape_: (6x49x49xf32, 0x6x2401xf32) <- (6x2401xf32, 3xi64)
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_41, full_int_array_1280), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x6x-1x-1xf32) <- (-1x6x-1x16xf32, -1x6x16x-1xf32)
        matmul_32 = paddle._C_ops.matmul(transpose_36, transpose_39, False, False)

        # pd_op.full: (1xf32) <- ()
        full_667 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x6x-1x-1xf32) <- (-1x6x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_32, full_667, float('0'), True)

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x-1x-1xf32, 6x49x49xf32)
        add_7 = paddle._C_ops.add(scale_2, reshape__82)

        # pd_op.softmax_: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax__5 = paddle._C_ops.softmax_(add_7, -1)

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x-1x32xf32)
        matmul_33 = paddle._C_ops.matmul(softmax__5, transpose_38, False, False)

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_668 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_42 = [slice_594, slice_595, full_668]

        # pd_op.reshape_: (-1x-1x192xf32, 0x-1x49x6x32xf32) <- (-1x49x6x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x192xf32) <- (-1x-1x192xf32)
        hardswish_13 = paddle._C_ops.hardswish(reshape__84)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x192xf32, 192x256xf32)
        matmul_34 = paddle._C_ops.matmul(hardswish_13, parameter_142, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_44, flatten_45 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_34, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_44, parameter_143, parameter_144, parameter_145, parameter_146, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_29 = paddle._C_ops.shape(matmul_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1281 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1282 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_650 = paddle._C_ops.slice(shape_29, [0], full_int_array_1281, full_int_array_1282, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1283 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1284 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_651 = paddle._C_ops.slice(shape_29, [0], full_int_array_1283, full_int_array_1284, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_669 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_650, slice_651, full_669]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__156, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_8 = paddle._C_ops.add(add_6, reshape__86)

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x256xf32, 256x512xf32)
        matmul_35 = paddle._C_ops.matmul(add_8, parameter_147, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_46, flatten_47 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_35, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_46, parameter_148, parameter_149, parameter_150, parameter_151, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_30 = paddle._C_ops.shape(matmul_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1285 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1286 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_652 = paddle._C_ops.slice(shape_30, [0], full_int_array_1285, full_int_array_1286, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1287 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1288 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_653 = paddle._C_ops.slice(shape_30, [0], full_int_array_1287, full_int_array_1288, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_670 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_44 = [slice_652, slice_653, full_670]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__162, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x512xf32) <- (-1x-1x512xf32)
        hardswish_14 = paddle._C_ops.hardswish(reshape__88)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_36 = paddle._C_ops.matmul(hardswish_14, parameter_152, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_48, flatten_49 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_36, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_48, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_31 = paddle._C_ops.shape(matmul_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1289 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1290 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_654 = paddle._C_ops.slice(shape_31, [0], full_int_array_1289, full_int_array_1290, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1291 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1292 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_655 = paddle._C_ops.slice(shape_31, [0], full_int_array_1291, full_int_array_1292, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_671 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_45 = [slice_654, slice_655, full_671]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__168, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_9 = paddle._C_ops.add(add_8, reshape__90)

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_32 = paddle._C_ops.shape(add_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1293 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1294 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_656 = paddle._C_ops.slice(shape_32, [0], full_int_array_1293, full_int_array_1294, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1295 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1296 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_657 = paddle._C_ops.slice(shape_32, [0], full_int_array_1295, full_int_array_1296, [1], [0])

        # pd_op.matmul: (-1x-1x1280xf32) <- (-1x-1x256xf32, 256x1280xf32)
        matmul_37 = paddle._C_ops.matmul(add_9, parameter_157, False, False)

        # pd_op.flatten: (-1x1280xf32, None) <- (-1x-1x1280xf32)
        flatten_50, flatten_51 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_37, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x1280xf32, 1280xf32, 1280xf32, xf32, xf32, None) <- (-1x1280xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_50, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x1280xf32)
        shape_33 = paddle._C_ops.shape(matmul_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1297 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1298 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_658 = paddle._C_ops.slice(shape_33, [0], full_int_array_1297, full_int_array_1298, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1299 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1300 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_659 = paddle._C_ops.slice(shape_33, [0], full_int_array_1299, full_int_array_1300, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_672 = paddle._C_ops.full([1], float('1280'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_658, slice_659, full_672]

        # pd_op.reshape_: (-1x-1x1280xf32, 0x-1x1280xf32) <- (-1x1280xf32, [1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__174, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_673 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_674 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_47 = [slice_656, slice_657, full_673, full_674]

        # pd_op.reshape_: (-1x-1x16x-1xf32, 0x-1x-1x1280xf32) <- (-1x-1x1280xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__92, [x.reshape([]) for x in combine_47]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1301 = [16, 64]

        # pd_op.full: (1xi32) <- ()
        full_675 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x16x-1xf32, -1x-1x16x-1xf32]) <- (-1x-1x16x-1xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(reshape__94, full_int_array_1301, full_675)

        # builtin.slice: (-1x-1x16x-1xf32) <- ([-1x-1x16x-1xf32, -1x-1x16x-1xf32])
        slice_660 = split_6[0]

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x-1x16x-1xf32)
        transpose_43 = paddle._C_ops.transpose(slice_660, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x16x-1xf32) <- ([-1x-1x16x-1xf32, -1x-1x16x-1xf32])
        slice_661 = split_6[1]

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x-1x16x-1xf32)
        transpose_44 = paddle._C_ops.transpose(slice_661, [0, 2, 1, 3])

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_34 = paddle._C_ops.shape(add_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1302 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1303 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_662 = paddle._C_ops.slice(shape_34, [0], full_int_array_1302, full_int_array_1303, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_676 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_677 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_678 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_662, full_676, full_677, full_678]

        # pd_op.reshape_: (-1x7x7x256xf32, 0x-1x-1x256xf32) <- (-1x-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add_9, [x.reshape([]) for x in combine_48]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1304 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1305 = [7, 7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1306 = [2, 2]

        # pd_op.strided_slice: (-1x4x4x256xf32) <- (-1x7x7x256xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(reshape__96, [1, 2], full_int_array_1304, full_int_array_1305, full_int_array_1306)

        # pd_op.full: (1xi32) <- ()
        full_679 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_680 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_49 = [slice_662, full_679, full_680]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x4x4x256xf32) <- (-1x4x4x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(strided_slice_1, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_38 = paddle._C_ops.matmul(reshape__98, parameter_162, False, False)

        # pd_op.flatten: (-1x256xf32, None) <- (-1x-1x256xf32)
        flatten_52, flatten_53 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_38, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_52, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x256xf32)
        shape_35 = paddle._C_ops.shape(matmul_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1307 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1308 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_663 = paddle._C_ops.slice(shape_35, [0], full_int_array_1307, full_int_array_1308, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1309 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1310 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_664 = paddle._C_ops.slice(shape_35, [0], full_int_array_1309, full_int_array_1310, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_681 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_50 = [slice_663, slice_664, full_681]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x256xf32) <- (-1x256xf32, [1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__180, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_682 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_683 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_684 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_656, full_682, full_683, full_684]

        # pd_op.reshape_: (-1x16x16x16xf32, 0x-1x-1x256xf32) <- (-1x-1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__100, [x.reshape([]) for x in combine_51]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x16x16xf32) <- (-1x16x16x16xf32)
        transpose_45 = paddle._C_ops.transpose(reshape__102, [0, 2, 1, 3])

        # pd_op.transpose: (49x16xf32) <- (16x49xf32)
        transpose_46 = paddle._C_ops.transpose(parameter_167, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1311 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1312 = [1]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_665 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1311, full_int_array_1312, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_685 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_588 = paddle._C_ops.gather(transpose_46, slice_665, full_685)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1313 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1314 = [2]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_666 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1313, full_int_array_1314, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_686 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_589 = paddle._C_ops.gather(transpose_46, slice_666, full_686)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1315 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1316 = [3]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_667 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1315, full_int_array_1316, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_687 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_590 = paddle._C_ops.gather(transpose_46, slice_667, full_687)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1317 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1318 = [4]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_668 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1317, full_int_array_1318, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_688 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_591 = paddle._C_ops.gather(transpose_46, slice_668, full_688)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1319 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1320 = [5]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_669 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1319, full_int_array_1320, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_689 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_592 = paddle._C_ops.gather(transpose_46, slice_669, full_689)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1321 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1322 = [6]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_670 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1321, full_int_array_1322, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_690 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_593 = paddle._C_ops.gather(transpose_46, slice_670, full_690)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1323 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1324 = [7]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_671 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1323, full_int_array_1324, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_691 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_594 = paddle._C_ops.gather(transpose_46, slice_671, full_691)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1325 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1326 = [8]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_672 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1325, full_int_array_1326, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_692 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_595 = paddle._C_ops.gather(transpose_46, slice_672, full_692)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1327 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1328 = [9]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_673 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1327, full_int_array_1328, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_693 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_596 = paddle._C_ops.gather(transpose_46, slice_673, full_693)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1329 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1330 = [10]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_674 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1329, full_int_array_1330, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_694 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_597 = paddle._C_ops.gather(transpose_46, slice_674, full_694)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1331 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1332 = [11]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_675 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1331, full_int_array_1332, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_695 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_598 = paddle._C_ops.gather(transpose_46, slice_675, full_695)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1333 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1334 = [12]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_676 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1333, full_int_array_1334, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_696 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_599 = paddle._C_ops.gather(transpose_46, slice_676, full_696)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1335 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1336 = [13]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_677 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1335, full_int_array_1336, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_697 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_600 = paddle._C_ops.gather(transpose_46, slice_677, full_697)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1337 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1338 = [14]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_678 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1337, full_int_array_1338, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_698 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_601 = paddle._C_ops.gather(transpose_46, slice_678, full_698)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1339 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1340 = [15]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_679 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1339, full_int_array_1340, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_699 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_602 = paddle._C_ops.gather(transpose_46, slice_679, full_699)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1341 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1342 = [16]

        # pd_op.slice: (49xi64) <- (16x49xi64, 1xi64, 1xi64)
        slice_680 = paddle._C_ops.slice(parameter_168, [0], full_int_array_1341, full_int_array_1342, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_700 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (49x16xf32) <- (49x16xf32, 49xi64, 1xi32)
        gather_603 = paddle._C_ops.gather(transpose_46, slice_680, full_700)

        # builtin.combine: ([49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32]) <- (49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32)
        combine_52 = [gather_588, gather_589, gather_590, gather_591, gather_592, gather_593, gather_594, gather_595, gather_596, gather_597, gather_598, gather_599, gather_600, gather_601, gather_602, gather_603]

        # pd_op.full: (1xi32) <- ()
        full_701 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (784x16xf32) <- ([49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32, 49x16xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_52, full_701)

        # pd_op.transpose: (16x784xf32) <- (784x16xf32)
        transpose_47 = paddle._C_ops.transpose(concat_6, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1343 = [0, 16, 49]

        # pd_op.reshape_: (16x16x49xf32, 0x16x784xf32) <- (16x784xf32, 3xi64)
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_47, full_int_array_1343), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32)
        transpose_48 = paddle._C_ops.transpose(transpose_43, [0, 1, 3, 2])

        # pd_op.matmul: (-1x16x16x-1xf32) <- (-1x16x16x16xf32, -1x16x-1x-1xf32)
        matmul_39 = paddle._C_ops.matmul(transpose_45, transpose_48, False, False)

        # pd_op.full: (1xf32) <- ()
        full_702 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x16x16x-1xf32) <- (-1x16x16x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_39, full_702, float('0'), True)

        # pd_op.add: (-1x16x16x49xf32) <- (-1x16x16x-1xf32, 16x16x49xf32)
        add_10 = paddle._C_ops.add(scale_3, reshape__104)

        # pd_op.softmax_: (-1x16x16x49xf32) <- (-1x16x16x49xf32)
        softmax__6 = paddle._C_ops.softmax_(add_10, -1)

        # pd_op.matmul: (-1x16x16x-1xf32) <- (-1x16x16x49xf32, -1x16x-1x-1xf32)
        matmul_40 = paddle._C_ops.matmul(softmax__6, transpose_44, False, False)

        # pd_op.transpose: (-1x16x16x-1xf32) <- (-1x16x16x-1xf32)
        transpose_49 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_703 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_704 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_53 = [slice_656, full_703, full_704]

        # pd_op.reshape_: (-1x-1x1024xf32, 0x-1x16x16x-1xf32) <- (-1x16x16x-1xf32, [1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_49, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        hardswish_15 = paddle._C_ops.hardswish(reshape__106)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1024xf32, 1024x384xf32)
        matmul_41 = paddle._C_ops.matmul(hardswish_15, parameter_169, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_54, flatten_55 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_41, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_54, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_36 = paddle._C_ops.shape(matmul_41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1344 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1345 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_681 = paddle._C_ops.slice(shape_36, [0], full_int_array_1344, full_int_array_1345, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1346 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1347 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_682 = paddle._C_ops.slice(shape_36, [0], full_int_array_1346, full_int_array_1347, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_705 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_54 = [slice_681, slice_682, full_705]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__186, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_42 = paddle._C_ops.matmul(reshape__108, parameter_174, False, False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_56, flatten_57 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_42, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_56, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_37 = paddle._C_ops.shape(matmul_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1348 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1349 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_683 = paddle._C_ops.slice(shape_37, [0], full_int_array_1348, full_int_array_1349, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1350 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1351 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_684 = paddle._C_ops.slice(shape_37, [0], full_int_array_1350, full_int_array_1351, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_706 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_55 = [slice_683, slice_684, full_706]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__192, [x.reshape([]) for x in combine_55]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_16 = paddle._C_ops.hardswish(reshape__110)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_43 = paddle._C_ops.matmul(hardswish_16, parameter_179, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_58, flatten_59 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_43, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_58, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_38 = paddle._C_ops.shape(matmul_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1352 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1353 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_685 = paddle._C_ops.slice(shape_38, [0], full_int_array_1352, full_int_array_1353, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1354 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1355 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_686 = paddle._C_ops.slice(shape_38, [0], full_int_array_1354, full_int_array_1355, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_707 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_56 = [slice_685, slice_686, full_707]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__198, [x.reshape([]) for x in combine_56]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_11 = paddle._C_ops.add(reshape__108, reshape__112)

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_39 = paddle._C_ops.shape(add_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1356 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1357 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_687 = paddle._C_ops.slice(shape_39, [0], full_int_array_1356, full_int_array_1357, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1358 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1359 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_688 = paddle._C_ops.slice(shape_39, [0], full_int_array_1358, full_int_array_1359, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_44 = paddle._C_ops.matmul(add_11, parameter_184, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_60, flatten_61 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_44, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_60, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_40 = paddle._C_ops.shape(matmul_44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1360 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1361 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_689 = paddle._C_ops.slice(shape_40, [0], full_int_array_1360, full_int_array_1361, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1362 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1363 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_690 = paddle._C_ops.slice(shape_40, [0], full_int_array_1362, full_int_array_1363, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_708 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_57 = [slice_689, slice_690, full_708]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__204, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_709 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_710 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_687, slice_688, full_709, full_710]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__114, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1364 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_711 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_7 = paddle._C_ops.split(reshape__116, full_int_array_1364, full_711)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_691 = split_7[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_50 = paddle._C_ops.transpose(slice_691, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_692 = split_7[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_51 = paddle._C_ops.transpose(slice_692, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_693 = split_7[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_52 = paddle._C_ops.transpose(slice_693, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_53 = paddle._C_ops.transpose(transpose_51, [0, 1, 3, 2])

        # pd_op.transpose: (16x8xf32) <- (8x16xf32)
        transpose_54 = paddle._C_ops.transpose(parameter_189, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1365 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1366 = [1]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_694 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1365, full_int_array_1366, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_712 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_604 = paddle._C_ops.gather(transpose_54, slice_694, full_712)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1367 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1368 = [2]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_695 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1367, full_int_array_1368, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_713 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_605 = paddle._C_ops.gather(transpose_54, slice_695, full_713)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1369 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1370 = [3]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_696 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1369, full_int_array_1370, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_714 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_606 = paddle._C_ops.gather(transpose_54, slice_696, full_714)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1371 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1372 = [4]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_697 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1371, full_int_array_1372, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_715 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_607 = paddle._C_ops.gather(transpose_54, slice_697, full_715)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1373 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1374 = [5]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_698 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1373, full_int_array_1374, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_716 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_608 = paddle._C_ops.gather(transpose_54, slice_698, full_716)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1375 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1376 = [6]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_699 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1375, full_int_array_1376, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_717 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_609 = paddle._C_ops.gather(transpose_54, slice_699, full_717)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1377 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1378 = [7]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_700 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1377, full_int_array_1378, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_718 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_610 = paddle._C_ops.gather(transpose_54, slice_700, full_718)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1379 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1380 = [8]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_701 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1379, full_int_array_1380, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_719 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_611 = paddle._C_ops.gather(transpose_54, slice_701, full_719)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1381 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1382 = [9]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_702 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1381, full_int_array_1382, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_720 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_612 = paddle._C_ops.gather(transpose_54, slice_702, full_720)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1383 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1384 = [10]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_703 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1383, full_int_array_1384, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_721 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_613 = paddle._C_ops.gather(transpose_54, slice_703, full_721)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1385 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1386 = [11]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_704 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1385, full_int_array_1386, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_722 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_614 = paddle._C_ops.gather(transpose_54, slice_704, full_722)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1387 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1388 = [12]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_705 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1387, full_int_array_1388, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_723 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_615 = paddle._C_ops.gather(transpose_54, slice_705, full_723)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1389 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1390 = [13]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_706 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1389, full_int_array_1390, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_724 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_616 = paddle._C_ops.gather(transpose_54, slice_706, full_724)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1391 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1392 = [14]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_707 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1391, full_int_array_1392, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_725 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_617 = paddle._C_ops.gather(transpose_54, slice_707, full_725)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1393 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1394 = [15]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_708 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1393, full_int_array_1394, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_726 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_618 = paddle._C_ops.gather(transpose_54, slice_708, full_726)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1395 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1396 = [16]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_709 = paddle._C_ops.slice(parameter_190, [0], full_int_array_1395, full_int_array_1396, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_727 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_619 = paddle._C_ops.gather(transpose_54, slice_709, full_727)

        # builtin.combine: ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32]) <- (16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32)
        combine_59 = [gather_604, gather_605, gather_606, gather_607, gather_608, gather_609, gather_610, gather_611, gather_612, gather_613, gather_614, gather_615, gather_616, gather_617, gather_618, gather_619]

        # pd_op.full: (1xi32) <- ()
        full_728 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (256x8xf32) <- ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_59, full_728)

        # pd_op.transpose: (8x256xf32) <- (256x8xf32)
        transpose_55 = paddle._C_ops.transpose(concat_7, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1397 = [0, 16, 16]

        # pd_op.reshape_: (8x16x16xf32, 0x8x256xf32) <- (8x256xf32, 3xi64)
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_55, full_int_array_1397), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_45 = paddle._C_ops.matmul(transpose_50, transpose_53, False, False)

        # pd_op.full: (1xf32) <- ()
        full_729 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_45, full_729, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_12 = paddle._C_ops.add(scale_4, reshape__118)

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__7 = paddle._C_ops.softmax_(add_12, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_46 = paddle._C_ops.matmul(softmax__7, transpose_52, False, False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_56 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_730 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_60 = [slice_687, slice_688, full_730]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_56, [x.reshape([]) for x in combine_60]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_17 = paddle._C_ops.hardswish(reshape__120)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_47 = paddle._C_ops.matmul(hardswish_17, parameter_191, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_62, flatten_63 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_47, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_62, parameter_192, parameter_193, parameter_194, parameter_195, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_41 = paddle._C_ops.shape(matmul_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1398 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1399 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_710 = paddle._C_ops.slice(shape_41, [0], full_int_array_1398, full_int_array_1399, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1400 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1401 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_711 = paddle._C_ops.slice(shape_41, [0], full_int_array_1400, full_int_array_1401, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_731 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_61 = [slice_710, slice_711, full_731]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__210, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_13 = paddle._C_ops.add(add_11, reshape__122)

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_48 = paddle._C_ops.matmul(add_13, parameter_196, False, False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_64, flatten_65 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_48, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_64, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_42 = paddle._C_ops.shape(matmul_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1402 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1403 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_712 = paddle._C_ops.slice(shape_42, [0], full_int_array_1402, full_int_array_1403, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1404 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1405 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_713 = paddle._C_ops.slice(shape_42, [0], full_int_array_1404, full_int_array_1405, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_732 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_62 = [slice_712, slice_713, full_732]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__216, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_18 = paddle._C_ops.hardswish(reshape__124)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_49 = paddle._C_ops.matmul(hardswish_18, parameter_201, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_66, flatten_67 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_49, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_66, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_43 = paddle._C_ops.shape(matmul_49)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1406 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1407 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_714 = paddle._C_ops.slice(shape_43, [0], full_int_array_1406, full_int_array_1407, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1408 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1409 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_715 = paddle._C_ops.slice(shape_43, [0], full_int_array_1408, full_int_array_1409, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_733 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_63 = [slice_714, slice_715, full_733]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__222, [x.reshape([]) for x in combine_63]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_14 = paddle._C_ops.add(add_13, reshape__126)

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_44 = paddle._C_ops.shape(add_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1410 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1411 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_716 = paddle._C_ops.slice(shape_44, [0], full_int_array_1410, full_int_array_1411, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1412 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1413 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_717 = paddle._C_ops.slice(shape_44, [0], full_int_array_1412, full_int_array_1413, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_50 = paddle._C_ops.matmul(add_14, parameter_206, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_68, flatten_69 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_50, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_68, parameter_207, parameter_208, parameter_209, parameter_210, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_45 = paddle._C_ops.shape(matmul_50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1414 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1415 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_718 = paddle._C_ops.slice(shape_45, [0], full_int_array_1414, full_int_array_1415, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1416 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1417 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_719 = paddle._C_ops.slice(shape_45, [0], full_int_array_1416, full_int_array_1417, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_734 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_64 = [slice_718, slice_719, full_734]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__228, [x.reshape([]) for x in combine_64]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_735 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_736 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_716, slice_717, full_735, full_736]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__128, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1418 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_737 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_8 = paddle._C_ops.split(reshape__130, full_int_array_1418, full_737)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_720 = split_8[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_57 = paddle._C_ops.transpose(slice_720, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_721 = split_8[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_58 = paddle._C_ops.transpose(slice_721, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_722 = split_8[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_59 = paddle._C_ops.transpose(slice_722, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_60 = paddle._C_ops.transpose(transpose_58, [0, 1, 3, 2])

        # pd_op.transpose: (16x8xf32) <- (8x16xf32)
        transpose_61 = paddle._C_ops.transpose(parameter_211, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1419 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1420 = [1]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_723 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1419, full_int_array_1420, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_738 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_620 = paddle._C_ops.gather(transpose_61, slice_723, full_738)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1421 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1422 = [2]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_724 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1421, full_int_array_1422, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_739 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_621 = paddle._C_ops.gather(transpose_61, slice_724, full_739)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1423 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1424 = [3]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_725 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1423, full_int_array_1424, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_740 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_622 = paddle._C_ops.gather(transpose_61, slice_725, full_740)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1425 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1426 = [4]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_726 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1425, full_int_array_1426, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_741 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_623 = paddle._C_ops.gather(transpose_61, slice_726, full_741)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1427 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1428 = [5]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_727 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1427, full_int_array_1428, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_742 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_624 = paddle._C_ops.gather(transpose_61, slice_727, full_742)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1429 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1430 = [6]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_728 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1429, full_int_array_1430, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_743 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_625 = paddle._C_ops.gather(transpose_61, slice_728, full_743)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1431 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1432 = [7]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_729 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1431, full_int_array_1432, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_744 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_626 = paddle._C_ops.gather(transpose_61, slice_729, full_744)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1433 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1434 = [8]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_730 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1433, full_int_array_1434, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_745 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_627 = paddle._C_ops.gather(transpose_61, slice_730, full_745)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1435 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1436 = [9]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_731 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1435, full_int_array_1436, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_746 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_628 = paddle._C_ops.gather(transpose_61, slice_731, full_746)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1437 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1438 = [10]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_732 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1437, full_int_array_1438, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_747 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_629 = paddle._C_ops.gather(transpose_61, slice_732, full_747)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1439 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1440 = [11]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_733 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1439, full_int_array_1440, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_748 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_630 = paddle._C_ops.gather(transpose_61, slice_733, full_748)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1441 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1442 = [12]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_734 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1441, full_int_array_1442, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_749 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_631 = paddle._C_ops.gather(transpose_61, slice_734, full_749)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1443 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1444 = [13]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_735 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1443, full_int_array_1444, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_750 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_632 = paddle._C_ops.gather(transpose_61, slice_735, full_750)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1445 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1446 = [14]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_736 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1445, full_int_array_1446, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_751 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_633 = paddle._C_ops.gather(transpose_61, slice_736, full_751)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1447 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1448 = [15]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_737 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1447, full_int_array_1448, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_752 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_634 = paddle._C_ops.gather(transpose_61, slice_737, full_752)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1449 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1450 = [16]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_738 = paddle._C_ops.slice(parameter_212, [0], full_int_array_1449, full_int_array_1450, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_753 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_635 = paddle._C_ops.gather(transpose_61, slice_738, full_753)

        # builtin.combine: ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32]) <- (16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32)
        combine_66 = [gather_620, gather_621, gather_622, gather_623, gather_624, gather_625, gather_626, gather_627, gather_628, gather_629, gather_630, gather_631, gather_632, gather_633, gather_634, gather_635]

        # pd_op.full: (1xi32) <- ()
        full_754 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (256x8xf32) <- ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_66, full_754)

        # pd_op.transpose: (8x256xf32) <- (256x8xf32)
        transpose_62 = paddle._C_ops.transpose(concat_8, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1451 = [0, 16, 16]

        # pd_op.reshape_: (8x16x16xf32, 0x8x256xf32) <- (8x256xf32, 3xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_62, full_int_array_1451), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_51 = paddle._C_ops.matmul(transpose_57, transpose_60, False, False)

        # pd_op.full: (1xf32) <- ()
        full_755 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_51, full_755, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_15 = paddle._C_ops.add(scale_5, reshape__132)

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__8 = paddle._C_ops.softmax_(add_15, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax__8, transpose_59, False, False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_756 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_716, slice_717, full_756]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_63, [x.reshape([]) for x in combine_67]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_19 = paddle._C_ops.hardswish(reshape__134)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_53 = paddle._C_ops.matmul(hardswish_19, parameter_213, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_70, flatten_71 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_53, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_70, parameter_214, parameter_215, parameter_216, parameter_217, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_46 = paddle._C_ops.shape(matmul_53)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1452 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1453 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_739 = paddle._C_ops.slice(shape_46, [0], full_int_array_1452, full_int_array_1453, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1454 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1455 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_740 = paddle._C_ops.slice(shape_46, [0], full_int_array_1454, full_int_array_1455, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_757 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_739, slice_740, full_757]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__234, [x.reshape([]) for x in combine_68]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_16 = paddle._C_ops.add(add_14, reshape__136)

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_54 = paddle._C_ops.matmul(add_16, parameter_218, False, False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_72, flatten_73 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_54, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_72, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_47 = paddle._C_ops.shape(matmul_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1456 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1457 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_741 = paddle._C_ops.slice(shape_47, [0], full_int_array_1456, full_int_array_1457, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1458 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1459 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_742 = paddle._C_ops.slice(shape_47, [0], full_int_array_1458, full_int_array_1459, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_758 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_69 = [slice_741, slice_742, full_758]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__240, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_20 = paddle._C_ops.hardswish(reshape__138)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_55 = paddle._C_ops.matmul(hardswish_20, parameter_223, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_74, flatten_75 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_55, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_74, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_48 = paddle._C_ops.shape(matmul_55)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1460 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1461 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_743 = paddle._C_ops.slice(shape_48, [0], full_int_array_1460, full_int_array_1461, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1462 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1463 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_744 = paddle._C_ops.slice(shape_48, [0], full_int_array_1462, full_int_array_1463, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_759 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_70 = [slice_743, slice_744, full_759]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__246, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_17 = paddle._C_ops.add(add_16, reshape__140)

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_49 = paddle._C_ops.shape(add_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1464 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1465 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_745 = paddle._C_ops.slice(shape_49, [0], full_int_array_1464, full_int_array_1465, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1466 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1467 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_746 = paddle._C_ops.slice(shape_49, [0], full_int_array_1466, full_int_array_1467, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_56 = paddle._C_ops.matmul(add_17, parameter_228, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_76, flatten_77 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_56, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_76, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_50 = paddle._C_ops.shape(matmul_56)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1468 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1469 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_747 = paddle._C_ops.slice(shape_50, [0], full_int_array_1468, full_int_array_1469, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1470 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1471 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_748 = paddle._C_ops.slice(shape_50, [0], full_int_array_1470, full_int_array_1471, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_760 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_747, slice_748, full_760]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__252, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_761 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_762 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_745, slice_746, full_761, full_762]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__142, [x.reshape([]) for x in combine_72]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1472 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_763 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_9 = paddle._C_ops.split(reshape__144, full_int_array_1472, full_763)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_749 = split_9[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_64 = paddle._C_ops.transpose(slice_749, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_750 = split_9[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_65 = paddle._C_ops.transpose(slice_750, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_751 = split_9[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_66 = paddle._C_ops.transpose(slice_751, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_67 = paddle._C_ops.transpose(transpose_65, [0, 1, 3, 2])

        # pd_op.transpose: (16x8xf32) <- (8x16xf32)
        transpose_68 = paddle._C_ops.transpose(parameter_233, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1473 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1474 = [1]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_752 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1473, full_int_array_1474, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_764 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_636 = paddle._C_ops.gather(transpose_68, slice_752, full_764)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1475 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1476 = [2]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_753 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1475, full_int_array_1476, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_765 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_637 = paddle._C_ops.gather(transpose_68, slice_753, full_765)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1477 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1478 = [3]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_754 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1477, full_int_array_1478, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_766 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_638 = paddle._C_ops.gather(transpose_68, slice_754, full_766)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1479 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1480 = [4]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_755 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1479, full_int_array_1480, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_767 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_639 = paddle._C_ops.gather(transpose_68, slice_755, full_767)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1481 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1482 = [5]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_756 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1481, full_int_array_1482, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_768 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_640 = paddle._C_ops.gather(transpose_68, slice_756, full_768)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1483 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1484 = [6]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_757 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1483, full_int_array_1484, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_769 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_641 = paddle._C_ops.gather(transpose_68, slice_757, full_769)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1485 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1486 = [7]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_758 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1485, full_int_array_1486, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_770 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_642 = paddle._C_ops.gather(transpose_68, slice_758, full_770)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1487 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1488 = [8]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_759 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1487, full_int_array_1488, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_771 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_643 = paddle._C_ops.gather(transpose_68, slice_759, full_771)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1489 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1490 = [9]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_760 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1489, full_int_array_1490, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_772 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_644 = paddle._C_ops.gather(transpose_68, slice_760, full_772)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1491 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1492 = [10]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_761 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1491, full_int_array_1492, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_773 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_645 = paddle._C_ops.gather(transpose_68, slice_761, full_773)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1493 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1494 = [11]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_762 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1493, full_int_array_1494, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_774 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_646 = paddle._C_ops.gather(transpose_68, slice_762, full_774)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1495 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1496 = [12]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_763 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1495, full_int_array_1496, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_775 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_647 = paddle._C_ops.gather(transpose_68, slice_763, full_775)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1497 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1498 = [13]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_764 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1497, full_int_array_1498, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_776 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_648 = paddle._C_ops.gather(transpose_68, slice_764, full_776)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1499 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1500 = [14]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_765 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1499, full_int_array_1500, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_777 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_649 = paddle._C_ops.gather(transpose_68, slice_765, full_777)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1501 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1502 = [15]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_766 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1501, full_int_array_1502, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_778 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_650 = paddle._C_ops.gather(transpose_68, slice_766, full_778)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1503 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1504 = [16]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_767 = paddle._C_ops.slice(parameter_234, [0], full_int_array_1503, full_int_array_1504, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_779 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_651 = paddle._C_ops.gather(transpose_68, slice_767, full_779)

        # builtin.combine: ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32]) <- (16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32)
        combine_73 = [gather_636, gather_637, gather_638, gather_639, gather_640, gather_641, gather_642, gather_643, gather_644, gather_645, gather_646, gather_647, gather_648, gather_649, gather_650, gather_651]

        # pd_op.full: (1xi32) <- ()
        full_780 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (256x8xf32) <- ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_73, full_780)

        # pd_op.transpose: (8x256xf32) <- (256x8xf32)
        transpose_69 = paddle._C_ops.transpose(concat_9, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1505 = [0, 16, 16]

        # pd_op.reshape_: (8x16x16xf32, 0x8x256xf32) <- (8x256xf32, 3xi64)
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_69, full_int_array_1505), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_57 = paddle._C_ops.matmul(transpose_64, transpose_67, False, False)

        # pd_op.full: (1xf32) <- ()
        full_781 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_57, full_781, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_18 = paddle._C_ops.add(scale_6, reshape__146)

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__9 = paddle._C_ops.softmax_(add_18, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_58 = paddle._C_ops.matmul(softmax__9, transpose_66, False, False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_70 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_782 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_745, slice_746, full_782]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_70, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_21 = paddle._C_ops.hardswish(reshape__148)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_59 = paddle._C_ops.matmul(hardswish_21, parameter_235, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_78, flatten_79 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_59, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_78, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_51 = paddle._C_ops.shape(matmul_59)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1506 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1507 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_768 = paddle._C_ops.slice(shape_51, [0], full_int_array_1506, full_int_array_1507, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1508 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1509 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_769 = paddle._C_ops.slice(shape_51, [0], full_int_array_1508, full_int_array_1509, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_783 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_75 = [slice_768, slice_769, full_783]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__258, [x.reshape([]) for x in combine_75]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_19 = paddle._C_ops.add(add_17, reshape__150)

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_60 = paddle._C_ops.matmul(add_19, parameter_240, False, False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_80, flatten_81 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_60, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_80, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_52 = paddle._C_ops.shape(matmul_60)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1510 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1511 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_770 = paddle._C_ops.slice(shape_52, [0], full_int_array_1510, full_int_array_1511, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1512 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1513 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_771 = paddle._C_ops.slice(shape_52, [0], full_int_array_1512, full_int_array_1513, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_784 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_76 = [slice_770, slice_771, full_784]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__264, [x.reshape([]) for x in combine_76]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_22 = paddle._C_ops.hardswish(reshape__152)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_61 = paddle._C_ops.matmul(hardswish_22, parameter_245, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_82, flatten_83 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_61, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_82, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_53 = paddle._C_ops.shape(matmul_61)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1514 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1515 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_772 = paddle._C_ops.slice(shape_53, [0], full_int_array_1514, full_int_array_1515, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1516 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1517 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_773 = paddle._C_ops.slice(shape_53, [0], full_int_array_1516, full_int_array_1517, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_785 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_77 = [slice_772, slice_773, full_785]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__270, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_20 = paddle._C_ops.add(add_19, reshape__154)

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_54 = paddle._C_ops.shape(add_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1518 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1519 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_774 = paddle._C_ops.slice(shape_54, [0], full_int_array_1518, full_int_array_1519, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1520 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1521 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_775 = paddle._C_ops.slice(shape_54, [0], full_int_array_1520, full_int_array_1521, [1], [0])

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x384xf32, 384x512xf32)
        matmul_62 = paddle._C_ops.matmul(add_20, parameter_250, False, False)

        # pd_op.flatten: (-1x512xf32, None) <- (-1x-1x512xf32)
        flatten_84, flatten_85 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_62, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_84, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x512xf32)
        shape_55 = paddle._C_ops.shape(matmul_62)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1522 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1523 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_776 = paddle._C_ops.slice(shape_55, [0], full_int_array_1522, full_int_array_1523, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1524 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1525 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_777 = paddle._C_ops.slice(shape_55, [0], full_int_array_1524, full_int_array_1525, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_786 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_78 = [slice_776, slice_777, full_786]

        # pd_op.reshape_: (-1x-1x512xf32, 0x-1x512xf32) <- (-1x512xf32, [1xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__276, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_787 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_788 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_774, slice_775, full_787, full_788]

        # pd_op.reshape_: (-1x-1x8x64xf32, 0x-1x-1x512xf32) <- (-1x-1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(reshape__156, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1526 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_789 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32]) <- (-1x-1x8x64xf32, 3xi64, 1xi32)
        split_10 = paddle._C_ops.split(reshape__158, full_int_array_1526, full_789)

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_778 = split_10[0]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_71 = paddle._C_ops.transpose(slice_778, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x16xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_779 = split_10[1]

        # pd_op.transpose: (-1x8x-1x16xf32) <- (-1x-1x8x16xf32)
        transpose_72 = paddle._C_ops.transpose(slice_779, [0, 2, 1, 3])

        # builtin.slice: (-1x-1x8x32xf32) <- ([-1x-1x8x16xf32, -1x-1x8x16xf32, -1x-1x8x32xf32])
        slice_780 = split_10[2]

        # pd_op.transpose: (-1x8x-1x32xf32) <- (-1x-1x8x32xf32)
        transpose_73 = paddle._C_ops.transpose(slice_780, [0, 2, 1, 3])

        # pd_op.transpose: (-1x8x16x-1xf32) <- (-1x8x-1x16xf32)
        transpose_74 = paddle._C_ops.transpose(transpose_72, [0, 1, 3, 2])

        # pd_op.transpose: (16x8xf32) <- (8x16xf32)
        transpose_75 = paddle._C_ops.transpose(parameter_255, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1527 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1528 = [1]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_781 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1527, full_int_array_1528, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_790 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_652 = paddle._C_ops.gather(transpose_75, slice_781, full_790)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1529 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1530 = [2]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_782 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1529, full_int_array_1530, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_791 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_653 = paddle._C_ops.gather(transpose_75, slice_782, full_791)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1531 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1532 = [3]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_783 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1531, full_int_array_1532, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_792 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_654 = paddle._C_ops.gather(transpose_75, slice_783, full_792)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1533 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1534 = [4]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_784 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1533, full_int_array_1534, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_793 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_655 = paddle._C_ops.gather(transpose_75, slice_784, full_793)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1535 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1536 = [5]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_785 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1535, full_int_array_1536, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_794 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_656 = paddle._C_ops.gather(transpose_75, slice_785, full_794)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1537 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1538 = [6]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_786 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1537, full_int_array_1538, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_795 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_657 = paddle._C_ops.gather(transpose_75, slice_786, full_795)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1539 = [6]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1540 = [7]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_787 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1539, full_int_array_1540, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_796 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_658 = paddle._C_ops.gather(transpose_75, slice_787, full_796)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1541 = [7]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1542 = [8]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_788 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1541, full_int_array_1542, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_797 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_659 = paddle._C_ops.gather(transpose_75, slice_788, full_797)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1543 = [8]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1544 = [9]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_789 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1543, full_int_array_1544, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_798 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_660 = paddle._C_ops.gather(transpose_75, slice_789, full_798)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1545 = [9]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1546 = [10]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_790 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1545, full_int_array_1546, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_799 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_661 = paddle._C_ops.gather(transpose_75, slice_790, full_799)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1547 = [10]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1548 = [11]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_791 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1547, full_int_array_1548, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_800 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_662 = paddle._C_ops.gather(transpose_75, slice_791, full_800)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1549 = [11]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1550 = [12]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_792 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1549, full_int_array_1550, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_801 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_663 = paddle._C_ops.gather(transpose_75, slice_792, full_801)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1551 = [12]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1552 = [13]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_793 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1551, full_int_array_1552, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_802 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_664 = paddle._C_ops.gather(transpose_75, slice_793, full_802)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1553 = [13]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1554 = [14]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_794 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1553, full_int_array_1554, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_803 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_665 = paddle._C_ops.gather(transpose_75, slice_794, full_803)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1555 = [14]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1556 = [15]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_795 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1555, full_int_array_1556, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_804 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_666 = paddle._C_ops.gather(transpose_75, slice_795, full_804)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1557 = [15]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1558 = [16]

        # pd_op.slice: (16xi64) <- (16x16xi64, 1xi64, 1xi64)
        slice_796 = paddle._C_ops.slice(parameter_256, [0], full_int_array_1557, full_int_array_1558, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_805 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (16x8xf32) <- (16x8xf32, 16xi64, 1xi32)
        gather_667 = paddle._C_ops.gather(transpose_75, slice_796, full_805)

        # builtin.combine: ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32]) <- (16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32)
        combine_80 = [gather_652, gather_653, gather_654, gather_655, gather_656, gather_657, gather_658, gather_659, gather_660, gather_661, gather_662, gather_663, gather_664, gather_665, gather_666, gather_667]

        # pd_op.full: (1xi32) <- ()
        full_806 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (256x8xf32) <- ([16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32, 16x8xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_80, full_806)

        # pd_op.transpose: (8x256xf32) <- (256x8xf32)
        transpose_76 = paddle._C_ops.transpose(concat_10, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1559 = [0, 16, 16]

        # pd_op.reshape_: (8x16x16xf32, 0x8x256xf32) <- (8x256xf32, 3xi64)
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_76, full_int_array_1559), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x16xf32, -1x8x16x-1xf32)
        matmul_63 = paddle._C_ops.matmul(transpose_71, transpose_74, False, False)

        # pd_op.full: (1xf32) <- ()
        full_807 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_63, full_807, float('0'), True)

        # pd_op.add: (-1x8x16x16xf32) <- (-1x8x-1x-1xf32, 8x16x16xf32)
        add_21 = paddle._C_ops.add(scale_7, reshape__160)

        # pd_op.softmax_: (-1x8x16x16xf32) <- (-1x8x16x16xf32)
        softmax__10 = paddle._C_ops.softmax_(add_21, -1)

        # pd_op.matmul: (-1x8x16x32xf32) <- (-1x8x16x16xf32, -1x8x-1x32xf32)
        matmul_64 = paddle._C_ops.matmul(softmax__10, transpose_73, False, False)

        # pd_op.transpose: (-1x16x8x32xf32) <- (-1x8x16x32xf32)
        transpose_77 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_808 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_81 = [slice_774, slice_775, full_808]

        # pd_op.reshape_: (-1x-1x256xf32, 0x-1x16x8x32xf32) <- (-1x16x8x32xf32, [1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_77, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x256xf32) <- (-1x-1x256xf32)
        hardswish_23 = paddle._C_ops.hardswish(reshape__162)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x256xf32, 256x384xf32)
        matmul_65 = paddle._C_ops.matmul(hardswish_23, parameter_257, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_86, flatten_87 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_65, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_86, parameter_258, parameter_259, parameter_260, parameter_261, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_56 = paddle._C_ops.shape(matmul_65)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1560 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1561 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_797 = paddle._C_ops.slice(shape_56, [0], full_int_array_1560, full_int_array_1561, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1562 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1563 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_798 = paddle._C_ops.slice(shape_56, [0], full_int_array_1562, full_int_array_1563, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_809 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_82 = [slice_797, slice_798, full_809]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__282, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_22 = paddle._C_ops.add(add_20, reshape__164)

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x384xf32, 384x768xf32)
        matmul_66 = paddle._C_ops.matmul(add_22, parameter_262, False, False)

        # pd_op.flatten: (-1x768xf32, None) <- (-1x-1x768xf32)
        flatten_88, flatten_89 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_66, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x768xf32, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_88, parameter_263, parameter_264, parameter_265, parameter_266, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x768xf32)
        shape_57 = paddle._C_ops.shape(matmul_66)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1564 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1565 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_799 = paddle._C_ops.slice(shape_57, [0], full_int_array_1564, full_int_array_1565, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1566 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1567 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_800 = paddle._C_ops.slice(shape_57, [0], full_int_array_1566, full_int_array_1567, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_810 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_83 = [slice_799, slice_800, full_810]

        # pd_op.reshape_: (-1x-1x768xf32, 0x-1x768xf32) <- (-1x768xf32, [1xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__288, [x.reshape([]) for x in combine_83]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.hardswish: (-1x-1x768xf32) <- (-1x-1x768xf32)
        hardswish_24 = paddle._C_ops.hardswish(reshape__166)

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_67 = paddle._C_ops.matmul(hardswish_24, parameter_267, False, False)

        # pd_op.flatten: (-1x384xf32, None) <- (-1x-1x384xf32)
        flatten_90, flatten_91 = (lambda x, f: f(x))(paddle._C_ops.flatten(matmul_67, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(flatten_90, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (3xi32) <- (-1x-1x384xf32)
        shape_58 = paddle._C_ops.shape(matmul_67)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1568 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1569 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_801 = paddle._C_ops.slice(shape_58, [0], full_int_array_1568, full_int_array_1569, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1570 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1571 = [2]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_802 = paddle._C_ops.slice(shape_58, [0], full_int_array_1570, full_int_array_1571, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_811 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_84 = [slice_801, slice_802, full_811]

        # pd_op.reshape_: (-1x-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, [1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__294, [x.reshape([]) for x in combine_84]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_23 = paddle._C_ops.add(add_22, reshape__168)

        # pd_op.mean: (-1x384xf32) <- (-1x-1x384xf32)
        mean_0 = paddle._C_ops.mean(add_23, [1], False)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1572 = [-1, 384]

        # pd_op.reshape_: (-1x384xf32, 0x-1x384xf32) <- (-1x384xf32, 2xi64)
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(mean_0, full_int_array_1572), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__170, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x384xf32, 384x1000xf32)
        matmul_68 = paddle._C_ops.matmul(batch_norm__300, parameter_276, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__7 = paddle._C_ops.add_(matmul_68, parameter_277)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__11 = paddle._C_ops.softmax_(add__7, -1)
        return softmax__11



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_212, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_234, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, feed_0):
        return self.builtin_module_4725_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_31, parameter_28, parameter_30, parameter_29, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_48, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_76, parameter_80, parameter_77, parameter_79, parameter_78, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_141, parameter_142, parameter_146, parameter_143, parameter_145, parameter_144, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_190, parameter_191, parameter_195, parameter_192, parameter_194, parameter_193, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_212, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_234, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_4725_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.cast(paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'), 'int64'),
            # parameter_27
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256, 128], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.cast(paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'), 'int64'),
            # parameter_49
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256, 128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128, 640], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 196], dtype='int64'), 'int64'),
            # parameter_76
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([6, 49], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'), 'int64'),
            # parameter_98
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([6, 49], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'), 'int64'),
            # parameter_120
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([6, 49], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'), 'int64'),
            # parameter_142
            paddle.uniform([192, 256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512, 256], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256, 1280], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 2, 1, 0, 1, 2, 3, 4, 9, 8, 7, 8, 9, 10, 11, 16, 15, 14, 15, 16, 17, 18, 23, 22, 21, 22, 23, 24, 25, 30, 29, 28, 29, 30, 31, 32, 37, 36, 35, 36, 37, 38, 39, 44, 43, 42, 43, 44, 45, 46, 4, 3, 2, 1, 0, 1, 2, 11, 10, 9, 8, 7, 8, 9, 18, 17, 16, 15, 14, 15, 16, 25, 24, 23, 22, 21, 22, 23, 32, 31, 30, 29, 28, 29, 30, 39, 38, 37, 36, 35, 36, 37, 46, 45, 44, 43, 42, 43, 44, 6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 20, 19, 18, 17, 16, 15, 14, 27, 26, 25, 24, 23, 22, 21, 34, 33, 32, 31, 30, 29, 28, 41, 40, 39, 38, 37, 36, 35, 48, 47, 46, 45, 44, 43, 42, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 16, 15, 14, 15, 16, 17, 18, 9, 8, 7, 8, 9, 10, 11, 2, 1, 0, 1, 2, 3, 4, 9, 8, 7, 8, 9, 10, 11, 16, 15, 14, 15, 16, 17, 18, 23, 22, 21, 22, 23, 24, 25, 30, 29, 28, 29, 30, 31, 32, 18, 17, 16, 15, 14, 15, 16, 11, 10, 9, 8, 7, 8, 9, 4, 3, 2, 1, 0, 1, 2, 11, 10, 9, 8, 7, 8, 9, 18, 17, 16, 15, 14, 15, 16, 25, 24, 23, 22, 21, 22, 23, 32, 31, 30, 29, 28, 29, 30, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 20, 19, 18, 17, 16, 15, 14, 27, 26, 25, 24, 23, 22, 21, 34, 33, 32, 31, 30, 29, 28, 28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 29, 28, 29, 30, 31, 32, 23, 22, 21, 22, 23, 24, 25, 16, 15, 14, 15, 16, 17, 18, 9, 8, 7, 8, 9, 10, 11, 2, 1, 0, 1, 2, 3, 4, 9, 8, 7, 8, 9, 10, 11, 16, 15, 14, 15, 16, 17, 18, 32, 31, 30, 29, 28, 29, 30, 25, 24, 23, 22, 21, 22, 23, 18, 17, 16, 15, 14, 15, 16, 11, 10, 9, 8, 7, 8, 9, 4, 3, 2, 1, 0, 1, 2, 11, 10, 9, 8, 7, 8, 9, 18, 17, 16, 15, 14, 15, 16, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 20, 19, 18, 17, 16, 15, 14, 42, 43, 44, 45, 46, 47, 48, 35, 36, 37, 38, 39, 40, 41, 28, 29, 30, 31, 32, 33, 34, 21, 22, 23, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19, 20, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 44, 43, 42, 43, 44, 45, 46, 37, 36, 35, 36, 37, 38, 39, 30, 29, 28, 29, 30, 31, 32, 23, 22, 21, 22, 23, 24, 25, 16, 15, 14, 15, 16, 17, 18, 9, 8, 7, 8, 9, 10, 11, 2, 1, 0, 1, 2, 3, 4, 46, 45, 44, 43, 42, 43, 44, 39, 38, 37, 36, 35, 36, 37, 32, 31, 30, 29, 28, 29, 30, 25, 24, 23, 22, 21, 22, 23, 18, 17, 16, 15, 14, 15, 16, 11, 10, 9, 8, 7, 8, 9, 4, 3, 2, 1, 0, 1, 2, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16, 49]),
            # parameter_169
            paddle.uniform([1024, 384], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([8, 16], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16, 16]),
            # parameter_191
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([8, 16], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16, 16]),
            # parameter_213
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([8, 16], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16, 16]),
            # parameter_235
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([384, 512], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([8, 16], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16, 16]),
            # parameter_257
            paddle.uniform([256, 384], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([768, 384], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([384, 1000], dtype='float32', min=0, max=0.5),
            # parameter_277
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
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            # parameter_27
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256, 128], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
            # parameter_49
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256, 128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[128, 640], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[49, 196], dtype='int64'),
            # parameter_76
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[6, 49], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            # parameter_98
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[6, 49], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            # parameter_120
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[6, 49], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            # parameter_142
            paddle.static.InputSpec(shape=[192, 256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512, 256], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256, 1280], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[16, 49], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[16, 49], dtype='int64'),
            # parameter_169
            paddle.static.InputSpec(shape=[1024, 384], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            # parameter_191
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            # parameter_213
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            # parameter_235
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[384, 512], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[16, 16], dtype='int64'),
            # parameter_257
            paddle.static.InputSpec(shape=[256, 384], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[768, 384], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[384, 1000], dtype='float32'),
            # parameter_277
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