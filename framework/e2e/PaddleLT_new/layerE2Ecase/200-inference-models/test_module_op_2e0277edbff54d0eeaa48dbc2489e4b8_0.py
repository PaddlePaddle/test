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
    return [376][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_574_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, feed_0):

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x3x-1x-1xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x32x-1x-1xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_0, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__30, batch_norm__24)

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_4, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_5, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_1 = paddle._C_ops.add(relu_4, batch_norm__42)

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_6, parameter_40, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_7, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu_6, full_int_array_1, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_1, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__60, batch_norm__54)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_8, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__66)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_9, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_3 = paddle._C_ops.add(relu_8, batch_norm__72)

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu_10, parameter_65, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__78)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_11, parameter_70, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_10, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_4 = paddle._C_ops.add(batch_norm__90, batch_norm__84)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu_12, parameter_80, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_13, parameter_85, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_5 = paddle._C_ops.add(relu_12, batch_norm__102)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu_14, parameter_90, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu_15, parameter_95, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_14, parameter_100, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_6 = paddle._C_ops.add(batch_norm__120, batch_norm__114)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu_16, parameter_105, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_17, parameter_110, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_7 = paddle._C_ops.add(relu_16, batch_norm__132)

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_7)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu_18, full_int_array_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(pool2d_2, parameter_115, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_116, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_23, reshape_0)

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_117, parameter_118, parameter_119, parameter_120, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(relu_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_4, full_int_array_5, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__0 = paddle._C_ops.cast_(slice_0, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x1x1xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(relu__0, cast__0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [2, 2]

        # pd_op.pool2d: (-1x512x2x2xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu_18, full_int_array_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x2x2xf32) <- (-1x512x2x2xf32, 128x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(pool2d_3, parameter_121, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_122, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x2x2xf32) <- (-1x128x2x2xf32, 1x128x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_24, reshape_2)

        # pd_op.batch_norm_: (-1x128x2x2xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x2x2xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__1, parameter_123, parameter_124, parameter_125, parameter_126, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x2x2xf32) <- (-1x128x2x2xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(relu_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_8, full_int_array_9, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__1 = paddle._C_ops.cast_(slice_1, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x2x2xf32, 2xi32, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(relu__1, cast__1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [3, 3]

        # pd_op.pool2d: (-1x512x3x3xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu_18, full_int_array_10, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x3x3xf32) <- (-1x512x3x3xf32, 128x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(pool2d_4, parameter_127, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_128, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x3x3xf32) <- (-1x128x3x3xf32, 1x128x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_25, reshape_4)

        # pd_op.batch_norm_: (-1x128x3x3xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x3x3xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__2, parameter_129, parameter_130, parameter_131, parameter_132, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x3x3xf32) <- (-1x128x3x3xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(relu_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_12, full_int_array_13, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__2 = paddle._C_ops.cast_(slice_2, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x3x3xf32, 2xi32, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(relu__2, cast__2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [6, 6]

        # pd_op.pool2d: (-1x512x6x6xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(relu_18, full_int_array_14, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x6x6xf32) <- (-1x512x6x6xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(pool2d_5, parameter_133, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_134, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x6x6xf32) <- (-1x128x6x6xf32, 1x128x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_26, reshape_6)

        # pd_op.batch_norm_: (-1x128x6x6xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x6x6xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__3, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x6x6xf32) <- (-1x128x6x6xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(relu_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_16, full_int_array_17, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__3 = paddle._C_ops.cast_(slice_3, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x6x6xf32, 2xi32, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(relu__3, cast__3, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # builtin.combine: ([-1x512x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32]) <- (-1x512x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32)
        combine_0 = [relu_18, bilinear_interp_3, bilinear_interp_2, bilinear_interp_1, bilinear_interp_0]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x-1x-1xf32) <- ([-1x512x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x1024x-1x-1xf32, 128x1024x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(concat_0, parameter_139, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_140, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_27, reshape_8)

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_8, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu_14, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_146, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_28, reshape_10)

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_9, parameter_147, parameter_148, parameter_149, parameter_150, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_4 = paddle._C_ops.shape(relu_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_20, full_int_array_21, [1], [])

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu_20, parameter_151, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu_19, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_0 = paddle._C_ops.cast(slice_4, paddle.int32)

        # pd_op.bilinear_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(conv2d_30, cast_0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # builtin.combine: ([-1x64x-1x-1xf32, -1x64x-1x-1xf32]) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        combine_1 = [bilinear_interp_4, conv2d_29]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x-1x-1xf32) <- ([-1x64x-1x-1xf32, -1x64x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x128x-1x-1xf32, 2x128x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(concat_1, parameter_153, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [-1]

        # pd_op.strided_slice: (2xi32) <- (2xi32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(slice_4, [0], full_int_array_22, full_int_array_23, full_int_array_24)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 1, 1, -1]

        # pd_op.reshape_: (1x1x1x2xi32, 0x2xi32) <- (2xi32, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(strided_slice_0, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(slice_4, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_0 = paddle._C_ops.linspace(full_2, full_3, slice_5, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_0, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(slice_4, [0], full_int_array_29, full_int_array_30, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_2 = [slice_6]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_0 = paddle._C_ops.tile(reshape__2, [x.reshape([]) for x in combine_2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(slice_4, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_1 = paddle._C_ops.linspace(full_4, full_5, slice_7, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_33 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_1, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(slice_4, [0], full_int_array_34, full_int_array_35, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_3 = [slice_8]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_1 = paddle._C_ops.tile(reshape__4, [x.reshape([]) for x in combine_3])

        # pd_op.transpose: (-1x-1xf32) <- (-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(tile_1, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_0, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(tile_0, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_4 = [unsqueeze__0, unsqueeze__2]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_4, full_6)

        # pd_op.transpose: (-1x-1x-1x2xf32) <- (-1x2x-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(conv2d_31, [0, 2, 3, 1])

        # pd_op.cast: (1x1x1x2xf32) <- (1x1x1x2xi32)
        cast_1 = paddle._C_ops.cast(reshape__0, paddle.float32)

        # pd_op.memcpy_h2d: (1x1x1x2xf32) <- (1x1x1x2xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(cast_1, 1)

        # pd_op.divide: (-1x-1x-1x2xf32) <- (-1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_0 = paddle._C_ops.divide(transpose_1, memcpy_h2d_0)

        # pd_op.add: (-1x-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x-1x2xf32)
        add_10 = paddle._C_ops.add(concat_2, divide_0)

        # pd_op.grid_sample: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x-1x-1x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(relu_19, add_10, 'bilinear', 'zeros', True)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_11 = paddle._C_ops.add(relu_20, grid_sample_0)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(add_11, parameter_154, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__174)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu_10, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_160, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_33, reshape_12)

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_12, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_5 = paddle._C_ops.shape(relu_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_5, [0], full_int_array_39, full_int_array_40, [1], [])

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu_22, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(add_11, parameter_166, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_2 = paddle._C_ops.cast(slice_9, paddle.int32)

        # pd_op.bilinear_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(conv2d_35, cast_2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # builtin.combine: ([-1x64x-1x-1xf32, -1x64x-1x-1xf32]) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        combine_5 = [bilinear_interp_5, conv2d_34]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x-1x-1xf32) <- ([-1x64x-1x-1xf32, -1x64x-1x-1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_5, full_7)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x128x-1x-1xf32, 2x128x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(concat_3, parameter_167, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [-1]

        # pd_op.strided_slice: (2xi32) <- (2xi32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(slice_9, [0], full_int_array_41, full_int_array_42, full_int_array_43)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 1, 1, -1]

        # pd_op.reshape_: (1x1x1x2xi32, 0x2xi32) <- (2xi32, 4xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(strided_slice_1, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(slice_9, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_2 = paddle._C_ops.linspace(full_8, full_9, slice_10, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_47 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_2, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(slice_9, [0], full_int_array_48, full_int_array_49, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_6 = [slice_11]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_2 = paddle._C_ops.tile(reshape__8, [x.reshape([]) for x in combine_6])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(slice_9, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_3 = paddle._C_ops.linspace(full_10, full_11, slice_12, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_52 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_3, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(slice_9, [0], full_int_array_53, full_int_array_54, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_7 = [slice_13]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_3 = paddle._C_ops.tile(reshape__10, [x.reshape([]) for x in combine_7])

        # pd_op.transpose: (-1x-1xf32) <- (-1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(tile_3, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_2, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(tile_2, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_8 = [unsqueeze__4, unsqueeze__6]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_8, full_12)

        # pd_op.transpose: (-1x-1x-1x2xf32) <- (-1x2x-1x-1xf32)
        transpose_3 = paddle._C_ops.transpose(conv2d_36, [0, 2, 3, 1])

        # pd_op.cast: (1x1x1x2xf32) <- (1x1x1x2xi32)
        cast_3 = paddle._C_ops.cast(reshape__6, paddle.float32)

        # pd_op.memcpy_h2d: (1x1x1x2xf32) <- (1x1x1x2xf32)
        memcpy_h2d_1 = paddle._C_ops.memcpy_h2d(cast_3, 1)

        # pd_op.divide: (-1x-1x-1x2xf32) <- (-1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_1 = paddle._C_ops.divide(transpose_3, memcpy_h2d_1)

        # pd_op.add: (-1x-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x-1x2xf32)
        add_13 = paddle._C_ops.add(concat_4, divide_1)

        # pd_op.grid_sample: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x-1x-1x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(add_11, add_13, 'bilinear', 'zeros', True)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_14 = paddle._C_ops.add(relu_22, grid_sample_1)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(add_14, parameter_168, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_169, parameter_170, parameter_171, parameter_172, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__186)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu_6, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_174, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_38, reshape_14)

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add_15, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__192)

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_6 = paddle._C_ops.shape(relu_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_6, [0], full_int_array_58, full_int_array_59, [1], [])

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu_24, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x128x-1x-1xf32, 64x128x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(add_14, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_4 = paddle._C_ops.cast(slice_14, paddle.int32)

        # pd_op.bilinear_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_6 = paddle._C_ops.bilinear_interp(conv2d_40, cast_4, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # builtin.combine: ([-1x64x-1x-1xf32, -1x64x-1x-1xf32]) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        combine_9 = [bilinear_interp_6, conv2d_39]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x-1x-1xf32) <- ([-1x64x-1x-1xf32, -1x64x-1x-1xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_13)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x128x-1x-1xf32, 2x128x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(concat_5, parameter_181, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [-1]

        # pd_op.strided_slice: (2xi32) <- (2xi32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(slice_14, [0], full_int_array_60, full_int_array_61, full_int_array_62)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 1, 1, -1]

        # pd_op.reshape_: (1x1x1x2xi32, 0x2xi32) <- (2xi32, 4xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(strided_slice_2, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(slice_14, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_4 = paddle._C_ops.linspace(full_14, full_15, slice_15, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_66 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_4, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(slice_14, [0], full_int_array_67, full_int_array_68, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_10 = [slice_16]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_4 = paddle._C_ops.tile(reshape__14, [x.reshape([]) for x in combine_10])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [2]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(slice_14, [0], full_int_array_69, full_int_array_70, [1], [0])

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.linspace: (-1xf32) <- (1xf32, 1xf32, xi32)
        linspace_5 = paddle._C_ops.linspace(full_16, full_17, slice_17, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_71 = [-1, 1]

        # pd_op.reshape_: (-1x1xf32, 0x-1xf32) <- (-1xf32, 2xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(linspace_5, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (xi32) <- (2xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(slice_14, [0], full_int_array_72, full_int_array_73, [1], [0])

        # builtin.combine: ([xi32]) <- (xi32)
        combine_11 = [slice_18]

        # pd_op.tile: (-1x-1xf32) <- (-1x1xf32, [xi32])
        tile_5 = paddle._C_ops.tile(reshape__16, [x.reshape([]) for x in combine_11])

        # pd_op.transpose: (-1x-1xf32) <- (-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(tile_5, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_4, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [2]

        # pd_op.unsqueeze_: (-1x-1x1xf32, None) <- (-1x-1xf32, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(tile_4, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_12 = [unsqueeze__8, unsqueeze__10]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_12, full_18)

        # pd_op.transpose: (-1x-1x-1x2xf32) <- (-1x2x-1x-1xf32)
        transpose_5 = paddle._C_ops.transpose(conv2d_41, [0, 2, 3, 1])

        # pd_op.cast: (1x1x1x2xf32) <- (1x1x1x2xi32)
        cast_5 = paddle._C_ops.cast(reshape__12, paddle.float32)

        # pd_op.memcpy_h2d: (1x1x1x2xf32) <- (1x1x1x2xf32)
        memcpy_h2d_2 = paddle._C_ops.memcpy_h2d(cast_5, 1)

        # pd_op.divide: (-1x-1x-1x2xf32) <- (-1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_2 = paddle._C_ops.divide(transpose_5, memcpy_h2d_2)

        # pd_op.add: (-1x-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x-1x2xf32)
        add_16 = paddle._C_ops.add(concat_6, divide_2)

        # pd_op.grid_sample: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x-1x-1x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(add_14, add_16, 'bilinear', 'zeros', True)

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_17 = paddle._C_ops.add(relu_24, grid_sample_2)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(add_17, parameter_182, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_183, parameter_184, parameter_185, parameter_186, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__198)

        # pd_op.shape: (4xi32) <- (-1x128x-1x-1xf32)
        shape_7 = paddle._C_ops.shape(relu_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_7, [0], full_int_array_76, full_int_array_77, [1], [])

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_6 = paddle._C_ops.cast(slice_19, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_7 = paddle._C_ops.bilinear_interp(relu_23, cast_6, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_7 = paddle._C_ops.cast(slice_19, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_8 = paddle._C_ops.bilinear_interp(relu_21, cast_7, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__4 = paddle._C_ops.cast_(slice_19, paddle.int32)

        # pd_op.bilinear_interp: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_9 = paddle._C_ops.bilinear_interp(relu_19, cast__4, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

        # builtin.combine: ([-1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32]) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32)
        combine_13 = [relu_25, bilinear_interp_7, bilinear_interp_8, bilinear_interp_9]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x-1x-1xf32) <- ([-1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32, -1x128x-1x-1xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_13, full_19)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(concat_7, parameter_187, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_188, parameter_189, parameter_190, parameter_191, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__204)

        # pd_op.conv2d: (-1x19x-1x-1xf32) <- (-1x128x-1x-1xf32, 19x128x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu_26, parameter_192, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_78 = [1, 19, 1, 1]

        # pd_op.reshape: (1x19x1x1xf32, 0x19xf32) <- (19xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_193, full_int_array_78), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x19x-1x-1xf32) <- (-1x19x-1x-1xf32, 1x19x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_44, reshape_16)

        # pd_op.shape: (4xi32) <- (-1x3x-1x-1xf32)
        shape_8 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_8, [0], full_int_array_79, full_int_array_80, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__5 = paddle._C_ops.cast_(slice_20, paddle.int32)

        # pd_op.bilinear_interp: (-1x19x-1x-1xf32) <- (-1x19x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_10 = paddle._C_ops.bilinear_interp(add_18, cast__5, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.full: (1xi64) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1x-1x-1xi32) <- (-1x19x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(bilinear_interp_10, full_20, False, False, paddle.int32)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1xi32) <- (-1x-1x-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(argmax_0, full_21, float('0'), True)
        return scale_0



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, feed_0):
        return self.builtin_module_574_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_128, parameter_132, parameter_129, parameter_131, parameter_130, parameter_133, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_574_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128, 1024, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([2, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([2, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([128, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([2, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([128, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([19, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([19], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128, 1024, 3, 3], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[2, 128, 3, 3], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[2, 128, 3, 3], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[2, 128, 3, 3], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[128, 512, 3, 3], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[19, 128, 1, 1], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[19], dtype='float32'),
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